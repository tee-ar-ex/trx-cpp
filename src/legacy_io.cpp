#include <trx/legacy_io.h>
#include <trx/trx.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include <thread>
#include <chrono>

namespace trx {
namespace legacy {

inline float swap_float(float f) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = f;
    u.i = __builtin_bswap32(u.i);
    return u.f;
}

inline int32_t swap_int32(int32_t i) {
    return __builtin_bswap32(i);
}



bool load_trx(const std::string &filename, Tractogram &tr) {
    try {
        auto trx = trx::AnyTrxFile::load(filename);
        size_t num_streamlines = trx.num_streamlines();
        size_t num_points = trx.num_vertices();
        
        tr.pts.resize(num_points * 3);
        tr.offsets.resize(num_streamlines + 1);
        tr.header = trx.header;

        // Load offsets
        if (!trx.offsets.empty()) {
            if (trx.offsets.dtype == "uint32") {
                auto mat = trx.offsets.as_matrix<uint32_t>();
                for (size_t i = 0; i <= num_streamlines; ++i) tr.offsets[i] = mat.data()[i];
            } else if (trx.offsets.dtype == "uint64") {
                auto mat = trx.offsets.as_matrix<uint64_t>();
                for (size_t i = 0; i <= num_streamlines; ++i) tr.offsets[i] = mat.data()[i];
            } else if (trx.offsets.dtype == "int32") {
                auto mat = trx.offsets.as_matrix<int32_t>();
                for (size_t i = 0; i <= num_streamlines; ++i) tr.offsets[i] = mat.data()[i];
            } else if (trx.offsets.dtype == "int64") {
                auto mat = trx.offsets.as_matrix<int64_t>();
                for (size_t i = 0; i <= num_streamlines; ++i) tr.offsets[i] = mat.data()[i];
            }
        }

        // Load positions quickly (bulk copy / fast casting)
        if (!trx.positions.empty()) {
            if (trx.positions.dtype == "float32") {
                auto mat = trx.positions.as_matrix<float>();
                std::memcpy(tr.pts.data(), mat.data(), num_points * 3 * sizeof(float));
            } else if (trx.positions.dtype == "float16") {
                auto mat = trx.positions.as_matrix<Eigen::half>();
                for (size_t i = 0; i < num_points * 3; ++i) {
                    tr.pts[i] = static_cast<float>(mat.data()[i]);
                }
            } else if (trx.positions.dtype == "float64") {
                auto mat = trx.positions.as_matrix<double>();
                for (size_t i = 0; i < num_points * 3; ++i) {
                    tr.pts[i] = static_cast<float>(mat.data()[i]);
                }
            }
        }
        
        tr.original_trx = std::make_shared<trx::AnyTrxFile>(std::move(trx));
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading TRX file: " << e.what() << std::endl;
        return false;
    }
}

bool load_trk(const std::string &filename, Tractogram &tr) {
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!f.read(buffer.data(), size)) return false;
    if (buffer.size() < 1000) return false;
    
    const TrkHeader* header = reinterpret_cast<const TrkHeader*>(buffer.data());
    if (std::string(header->magic_number, 5) != "TRACK") return false;
    
    int16_t n_scalars = header->nb_scalars_per_point;
    int16_t n_properties = header->nb_properties_per_streamline;
    
    // Store metadata
    tr.header = json11::Json::object {
        { "DIMENSIONS", json11::Json::array { header->dimensions[0], header->dimensions[1], header->dimensions[2] } },
        { "VOXEL_TO_RASMM", json11::Json::array {
            json11::Json::array { header->voxel_to_rasmm[0][0], header->voxel_to_rasmm[0][1], header->voxel_to_rasmm[0][2], header->voxel_to_rasmm[0][3] },
            json11::Json::array { header->voxel_to_rasmm[1][0], header->voxel_to_rasmm[1][1], header->voxel_to_rasmm[1][2], header->voxel_to_rasmm[1][3] },
            json11::Json::array { header->voxel_to_rasmm[2][0], header->voxel_to_rasmm[2][1], header->voxel_to_rasmm[2][2], header->voxel_to_rasmm[2][3] },
            json11::Json::array { header->voxel_to_rasmm[3][0], header->voxel_to_rasmm[3][1], header->voxel_to_rasmm[3][2], header->voxel_to_rasmm[3][3] }
        } }
    };
    
    tr.offsets.clear();
    tr.offsets.push_back(0);
    tr.pts.clear();
    
    size_t offset = 1000;
    while (offset + sizeof(int32_t) <= buffer.size()) {
        int32_t n_points = *reinterpret_cast<const int32_t*>(buffer.data() + offset);
        offset += sizeof(int32_t);
        
        tr.offsets.push_back(tr.offsets.back() + n_points);
        
        for (int32_t j = 0; j < n_points; ++j) {
            float x = *reinterpret_cast<const float*>(buffer.data() + offset);
            float y = *reinterpret_cast<const float*>(buffer.data() + offset + 4);
            float z = *reinterpret_cast<const float*>(buffer.data() + offset + 8);
            tr.pts.push_back(x);
            tr.pts.push_back(y);
            tr.pts.push_back(z);
            
            offset += (3 + n_scalars) * sizeof(float);
        }
        offset += n_properties * sizeof(float);
    }
    
    return true;
}

bool load_tck(const std::string &filename, Tractogram &tr) {
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!f.read(buffer.data(), size)) return false;
    
    std::string_view view(buffer.data(), buffer.size());
    size_t file_pos = view.find("file: . ");
    if (file_pos == std::string_view::npos) return false;
    size_t offset_pos = file_pos + 8;
    size_t offset_end = view.find_first_not_of("0123456789", offset_pos);
    if (offset_end == std::string_view::npos) return false;
    size_t offset = std::stoull(std::string(view.substr(offset_pos, offset_end - offset_pos)));
    
    if (offset >= buffer.size()) return false;
    
    const float* data = reinterpret_cast<const float*>(buffer.data() + offset);
    size_t num_floats = (buffer.size() - offset) / sizeof(float);
    size_t num_triplets = num_floats / 3;
    
    tr.offsets.clear();
    tr.offsets.push_back(0);
    tr.pts.clear();
    
    bool in_streamline = false;
    size_t current_pts = 0;
    
    for (size_t i = 0; i < num_triplets; ++i) {
        float x = data[i * 3];
        float y = data[i * 3 + 1];
        float z = data[i * 3 + 2];
        
        if (std::isinf(x) && std::isinf(y) && std::isinf(z)) {
            if (in_streamline) {
                tr.offsets.push_back(tr.offsets.back() + current_pts);
                current_pts = 0;
                in_streamline = false;
            }
            break;
        } else if (std::isnan(x) && std::isnan(y) && std::isnan(z)) {
            if (in_streamline) {
                tr.offsets.push_back(tr.offsets.back() + current_pts);
                current_pts = 0;
                in_streamline = false;
            }
        } else {
            in_streamline = true;
            tr.pts.push_back(x);
            tr.pts.push_back(y);
            tr.pts.push_back(z);
            current_pts++;
        }
    }
    
    if (in_streamline) {
        tr.offsets.push_back(tr.offsets.back() + current_pts);
    }
    
    return true;
}

bool load_vtk(const std::string &filename, Tractogram &tr) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) return false;

    std::string line;
    size_t num_points = 0;
    bool is_double = false;
    while (std::getline(f, line)) {
        if (line.rfind("POINTS ", 0) == 0) {
            size_t space1 = line.find(" ", 7);
            num_points = std::stoull(line.substr(7, space1 - 7));
            if (line.find("double", space1) != std::string::npos) {
                is_double = true;
            }
            break;
        }
    }
    if (num_points == 0) return false;

    tr.pts.resize(num_points * 3);
    if (is_double) {
        std::vector<double> dpts(num_points * 3);
        f.read(reinterpret_cast<char*>(dpts.data()), num_points * 3 * sizeof(double));
        for (size_t i = 0; i < num_points * 3; ++i) {
            uint64_t val;
            std::memcpy(&val, &dpts[i], 8);
            val = ((val & 0xFF00000000000000ULL) >> 56) | ((val & 0x00FF000000000000ULL) >> 40) |
                  ((val & 0x0000FF0000000000ULL) >> 24) | ((val & 0x000000FF00000000ULL) >> 8) |
                  ((val & 0x00000000FF000000ULL) << 8)  | ((val & 0x0000000000FF0000ULL) << 24) |
                  ((val & 0x000000000000FF00ULL) << 40) | ((val & 0x00000000000000FFULL) << 56);
            double swapped;
            std::memcpy(&swapped, &val, 8);
            tr.pts[i] = static_cast<float>(swapped);
        }
    } else {
        f.read(reinterpret_cast<char*>(tr.pts.data()), num_points * 3 * sizeof(float));
        for (size_t i = 0; i < num_points * 3; ++i) {
            tr.pts[i] = swap_float(tr.pts[i]);
        }
    }

    size_t num_streamlines = 0;
    while (std::getline(f, line)) {
        if (line.rfind("LINES ", 0) == 0) {
            num_streamlines = std::stoull(line.substr(6, line.find(" ", 6) - 6));
            break;
        }
    }
    if (num_streamlines == 0) return false;

    auto pos_before_offsets = f.tellg();
    std::getline(f, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    bool has_offsets = (line.rfind("OFFSETS", 0) == 0);
    bool is_int64 = (line.find("int64") != std::string::npos);

    if (has_offsets) {
        tr.offsets.resize(num_streamlines);
        for (size_t i = 0; i < num_streamlines; ++i) {
            if (is_int64) {
                uint64_t val;
                f.read(reinterpret_cast<char*>(&val), 8);
                val = ((val & 0xFF00000000000000ULL) >> 56) | ((val & 0x00FF000000000000ULL) >> 40) |
                      ((val & 0x0000FF0000000000ULL) >> 24) | ((val & 0x000000FF00000000ULL) >> 8) |
                      ((val & 0x00000000FF000000ULL) << 8)  | ((val & 0x0000000000FF0000ULL) << 24) |
                      ((val & 0x000000000000FF00ULL) << 40) | ((val & 0x00000000000000FFULL) << 56);
                tr.offsets[i] = val;
            } else {
                uint32_t val;
                f.read(reinterpret_cast<char*>(&val), 4);
                val = swap_int32(val);
                tr.offsets[i] = val;
            }
        }
        return true;
    }
    f.seekg(pos_before_offsets);

    tr.offsets.clear();
    tr.offsets.push_back(0);

    for (size_t i = 0; i < num_streamlines; ++i) {
        int32_t n_pts;
        f.read(reinterpret_cast<char*>(&n_pts), sizeof(int32_t));
        if (!f) break;
        n_pts = swap_int32(n_pts);
        if (n_pts == 0) continue;
        tr.offsets.push_back(tr.offsets.back() + n_pts);
        
        // Skip cell indices
        f.seekg(n_pts * sizeof(int32_t), std::ios::cur);
    }

    return true;
}

bool save_trx(const Tractogram &tr, const std::string &out_path) {
    try {
        if (tr.original_trx) {
            tr.original_trx->save(out_path, trx::TrxCompression::None);
            return true;
        }
        size_t nb_vertices = tr.pts.size() / 3;
        size_t nb_streamlines = tr.offsets.size() - 1;
        
        trx::TrxFile<float> trx(nb_vertices, nb_streamlines);
        
        // Copy positions
        std::memcpy(trx.streamlines->_data.data(), tr.pts.data(), tr.pts.size() * sizeof(float));
        
        // Copy offsets
        for (size_t i = 0; i <= nb_streamlines; ++i) {
            trx.streamlines->_offsets(i, 0) = tr.offsets[i];
        }
        
        // Compute lengths
        for (size_t i = 0; i < nb_streamlines; ++i) {
            trx.streamlines->_lengths(i, 0) = tr.offsets[i+1] - tr.offsets[i];
        }
        
        // Copy header
        trx.header = tr.header;
        
        trx.save(out_path, trx::TrxCompression::None);
        trx.close();
        
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error saving TRX file: " << e.what() << std::endl;
        return false;
    }
}

bool save_trk(const Tractogram &tr, const std::string &out_path, const std::string &original_filename) {
    std::ofstream f(out_path, std::ios::binary);
    if (!f.is_open()) return false;

    TrkHeader header;
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic_number, "TRACK", 5);
    
    // Default dimensions, voxel sizes and affine
    header.dimensions[0] = 256; header.dimensions[1] = 256; header.dimensions[2] = 256;
    header.voxel_sizes[0] = 1.0f; header.voxel_sizes[1] = 1.0f; header.voxel_sizes[2] = 1.0f;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            header.voxel_to_rasmm[r][c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    // Attempt to extract from JSON header
    if (tr.header["DIMENSIONS"].is_array()) {
        auto dims = tr.header["DIMENSIONS"].array_items();
        if (dims.size() >= 3) {
            header.dimensions[0] = static_cast<int16_t>(dims[0].number_value());
            header.dimensions[1] = static_cast<int16_t>(dims[1].number_value());
            header.dimensions[2] = static_cast<int16_t>(dims[2].number_value());
        }
    }
    if (tr.header["VOXEL_TO_RASMM"].is_array()) {
        auto rows = tr.header["VOXEL_TO_RASMM"].array_items();
        if (rows.size() >= 4) {
            float vox_to_ras[4][4];
            for (int r = 0; r < 4; ++r) {
                auto cols = rows[r].array_items();
                if (cols.size() >= 4) {
                    for (int c = 0; c < 4; ++c) {
                        vox_to_ras[r][c] = static_cast<float>(cols[c].number_value());
                        header.voxel_to_rasmm[r][c] = vox_to_ras[r][c];
                    }
                }
            }
            header.voxel_sizes[0] = std::sqrt(vox_to_ras[0][0]*vox_to_ras[0][0] + vox_to_ras[1][0]*vox_to_ras[1][0] + vox_to_ras[2][0]*vox_to_ras[2][0]);
            header.voxel_sizes[1] = std::sqrt(vox_to_ras[0][1]*vox_to_ras[0][1] + vox_to_ras[1][1]*vox_to_ras[1][1] + vox_to_ras[2][1]*vox_to_ras[2][1]);
            header.voxel_sizes[2] = std::sqrt(vox_to_ras[0][2]*vox_to_ras[0][2] + vox_to_ras[1][2]*vox_to_ras[1][2] + vox_to_ras[2][2]*vox_to_ras[2][2]);
        }
    }

    std::memcpy(header.voxel_order, "RAS", 3);
    header.nb_streamlines = static_cast<int32_t>(tr.offsets.size() - 1);
    header.version = 2;
    header.hdr_size = 1000;

    f.write(reinterpret_cast<const char*>(&header), 1000);

    size_t num_streamlines = tr.offsets.size() - 1;
    std::vector<char> chunk;
    chunk.reserve(4 * 1024 * 1024);

    for (size_t i = 0; i < num_streamlines; ++i) {
        size_t start = tr.offsets[i];
        size_t end = tr.offsets[i+1];
        int32_t n_pts = static_cast<int32_t>(end - start);

        // Push n_pts
        const char* p_n_pts = reinterpret_cast<const char*>(&n_pts);
        chunk.insert(chunk.end(), p_n_pts, p_n_pts + 4);

        // Push points
        for (size_t j = start; j < end; ++j) {
            float x = tr.pts[j*3];
            float y = tr.pts[j*3 + 1];
            float z = tr.pts[j*3 + 2];
            const char* px = reinterpret_cast<const char*>(&x);
            const char* py = reinterpret_cast<const char*>(&y);
            const char* pz = reinterpret_cast<const char*>(&z);
            chunk.insert(chunk.end(), px, px + 4);
            chunk.insert(chunk.end(), py, py + 4);
            chunk.insert(chunk.end(), pz, pz + 4);
        }

        if (chunk.size() >= 4000000) {
            f.write(chunk.data(), chunk.size());
            chunk.clear();
        }
    }

    if (!chunk.empty()) {
        f.write(chunk.data(), chunk.size());
    }

    return true;
}

bool save_tck(const Tractogram &tr, const std::string &out_path) {
    std::ofstream f(out_path, std::ios::binary);
    if (!f.is_open()) return false;

    size_t num_streamlines = tr.offsets.size() - 1;
    
    // Build TCK header
    std::string header;
    size_t offset = 80;
    while (true) {
        char buf[256];
        snprintf(buf, sizeof(buf), "mrtrix tracks\ncount: %010zu\ndatatype: Float32LE\nfile: . %zu\nEND\n", num_streamlines, offset);
        std::string h(buf);
        if (h.length() <= offset) {
            h.append(offset - h.length(), ' ');
            header = h;
            break;
        }
        offset = h.length();
    }
    f.write(header.data(), header.size());

    // Payload writing
    std::vector<float> chunk;
    chunk.reserve(1024 * 1024);

    for (size_t i = 0; i < num_streamlines; ++i) {
        size_t start = tr.offsets[i];
        size_t end = tr.offsets[i+1];
        
        for (size_t j = start; j < end; ++j) {
            chunk.push_back(tr.pts[j*3]);
            chunk.push_back(tr.pts[j*3 + 1]);
            chunk.push_back(tr.pts[j*3 + 2]);
            if (chunk.size() >= 1000000) {
                f.write(reinterpret_cast<const char*>(chunk.data()), chunk.size() * sizeof(float));
                chunk.clear();
            }
        }
        // Delimiter
        chunk.push_back(std::numeric_limits<float>::quiet_NaN());
        chunk.push_back(std::numeric_limits<float>::quiet_NaN());
        chunk.push_back(std::numeric_limits<float>::quiet_NaN());
    }

    // EOF Delimiter
    chunk.push_back(std::numeric_limits<float>::infinity());
    chunk.push_back(std::numeric_limits<float>::infinity());
    chunk.push_back(std::numeric_limits<float>::infinity());

    if (!chunk.empty()) {
        f.write(reinterpret_cast<const char*>(chunk.data()), chunk.size() * sizeof(float));
    }

    return true;
}

bool save_vtk(const Tractogram &tr, const std::string &out_path) {
    std::ofstream f(out_path, std::ios::binary);
    if (!f.is_open()) return false;

    size_t num_streamlines = tr.offsets.size() - 1;
    size_t num_points = tr.pts.size() / 3;

    // Write ASCII header
    char header[512];
    snprintf(header, sizeof(header), "# vtk DataFile Version 3.0\nvtk output\nBINARY\nDATASET POLYDATA\nPOINTS %zu float\n", num_points);
    f.write(header, std::strlen(header));

    // Write POINTS binary block (big-endian floats)
    std::vector<float> pts_buf;
    pts_buf.reserve(1024 * 1024);

    for (size_t i = 0; i < num_points * 3; ++i) {
        pts_buf.push_back(swap_float(tr.pts[i]));
        if (pts_buf.size() >= 1000000) {
            f.write(reinterpret_cast<const char*>(pts_buf.data()), pts_buf.size() * sizeof(float));
            pts_buf.clear();
        }
    }
    if (!pts_buf.empty()) {
        f.write(reinterpret_cast<const char*>(pts_buf.data()), pts_buf.size() * sizeof(float));
    }

    // Write LINES header
    size_t cell_array_size = num_streamlines + num_points;
    char lines_hdr[128];
    snprintf(lines_hdr, sizeof(lines_hdr), "LINES %zu %zu\n", num_streamlines, cell_array_size);
    f.write(lines_hdr, std::strlen(lines_hdr));

    // Write LINES binary block (big-endian int32)
    std::vector<int32_t> lines_buf;
    lines_buf.reserve(1024 * 1024);

    int32_t current_point_idx = 0;
    for (size_t i = 0; i < num_streamlines; ++i) {
        size_t start = tr.offsets[i];
        size_t end = tr.offsets[i+1];
        int32_t n_pts = static_cast<int32_t>(end - start);

        lines_buf.push_back(swap_int32(n_pts));
        for (int32_t j = 0; j < n_pts; ++j) {
            lines_buf.push_back(swap_int32(current_point_idx++));
        }

        if (lines_buf.size() >= 1000000) {
            f.write(reinterpret_cast<const char*>(lines_buf.data()), lines_buf.size() * sizeof(int32_t));
            lines_buf.clear();
        }
    }
    if (!lines_buf.empty()) {
        f.write(reinterpret_cast<const char*>(lines_buf.data()), lines_buf.size() * sizeof(int32_t));
    }

    return true;
}

} // namespace legacy
} // namespace trx
