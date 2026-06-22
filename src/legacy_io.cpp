#include <trx/legacy_io.h>
#include <trx/trx.h>
#include <Eigen/Dense>
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

inline int16_t swap_int16(int16_t val) {
    uint16_t uval = val;
    uval = (uval << 8) | (uval >> 8);
    return static_cast<int16_t>(uval);
}

inline int64_t swap_int64(int64_t val) {
    return __builtin_bswap64(val);
}

inline double swap_double(double d) {
    union {
        double d;
        uint64_t i;
    } u;
    u.d = d;
    u.i = __builtin_bswap64(u.i);
    return u.d;
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
            float raw_x = *reinterpret_cast<const float*>(buffer.data() + offset);
            float raw_y = *reinterpret_cast<const float*>(buffer.data() + offset + 4);
            float raw_z = *reinterpret_cast<const float*>(buffer.data() + offset + 8);

            float vx = header->voxel_sizes[0] > 0 ? header->voxel_sizes[0] : 1.0f;
            float vy = header->voxel_sizes[1] > 0 ? header->voxel_sizes[1] : 1.0f;
            float vz = header->voxel_sizes[2] > 0 ? header->voxel_sizes[2] : 1.0f;

            float cx = (raw_x / vx) - 0.5f;
            float cy = (raw_y / vy) - 0.5f;
            float cz = (raw_z / vz) - 0.5f;

            float x = cx * header->voxel_to_rasmm[0][0] + cy * header->voxel_to_rasmm[0][1] + cz * header->voxel_to_rasmm[0][2] + header->voxel_to_rasmm[0][3];
            float y = cx * header->voxel_to_rasmm[1][0] + cy * header->voxel_to_rasmm[1][1] + cz * header->voxel_to_rasmm[1][2] + header->voxel_to_rasmm[1][3];
            float z = cx * header->voxel_to_rasmm[2][0] + cy * header->voxel_to_rasmm[2][1] + cz * header->voxel_to_rasmm[2][2] + header->voxel_to_rasmm[2][3];

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

bool load_nifti_header(const std::string &ref_path, json11::Json &out_header) {
    std::ifstream f(ref_path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open reference NIfTI file: " << ref_path << "\n";
        return false;
    }
    char buf[540];
    f.read(buf, 540);
    if (f.gcount() < 348) {
        std::cerr << "Error: Invalid NIfTI file (too small)\n";
        return false;
    }
    
    int32_t sizeof_hdr;
    std::memcpy(&sizeof_hdr, buf, sizeof(int32_t));
    
    bool swap_endian = false;
    if (sizeof_hdr == 1543569408 || sizeof_hdr == 469893120) {
        swap_endian = true;
        sizeof_hdr = swap_int32(sizeof_hdr);
    }
    
    std::vector<float> dims(3);
    float dx, dy, dz, qfac;
    int sform_code, qform_code;
    float srow_x[4], srow_y[4], srow_z[4];
    float qoffset_x, qoffset_y, qoffset_z, b, c, d;

    if (sizeof_hdr == 348) { // NIfTI-1
        int16_t dim[8];
        std::memcpy(dim, buf + 40, 8 * sizeof(int16_t));
        if (swap_endian) for(int i=0; i<8; i++) dim[i] = swap_int16(dim[i]);
        dims[0] = dim[1]; dims[1] = dim[2]; dims[2] = dim[3];

        float pixdim[8];
        std::memcpy(pixdim, buf + 76, 8 * sizeof(float));
        if (swap_endian) for(int i=0; i<8; i++) pixdim[i] = swap_float(pixdim[i]);
        qfac = (pixdim[0] == 0.0f) ? 1.0f : pixdim[0];
        dx = pixdim[1]; dy = pixdim[2]; dz = pixdim[3];

        int16_t sform16, qform16;
        std::memcpy(&qform16, buf + 252, sizeof(int16_t));
        std::memcpy(&sform16, buf + 254, sizeof(int16_t));
        if (swap_endian) { qform16 = swap_int16(qform16); sform16 = swap_int16(sform16); }
        qform_code = qform16; sform_code = sform16;

        std::memcpy(&b, buf + 256, sizeof(float));
        std::memcpy(&c, buf + 260, sizeof(float));
        std::memcpy(&d, buf + 264, sizeof(float));
        std::memcpy(&qoffset_x, buf + 268, sizeof(float));
        std::memcpy(&qoffset_y, buf + 272, sizeof(float));
        std::memcpy(&qoffset_z, buf + 276, sizeof(float));
        if (swap_endian) {
            b = swap_float(b); c = swap_float(c); d = swap_float(d);
            qoffset_x = swap_float(qoffset_x); qoffset_y = swap_float(qoffset_y); qoffset_z = swap_float(qoffset_z);
        }
        
        std::memcpy(srow_x, buf + 280, 4 * sizeof(float));
        std::memcpy(srow_y, buf + 296, 4 * sizeof(float));
        std::memcpy(srow_z, buf + 312, 4 * sizeof(float));
        if (swap_endian) {
            for(int i=0; i<4; i++) {
                srow_x[i] = swap_float(srow_x[i]);
                srow_y[i] = swap_float(srow_y[i]);
                srow_z[i] = swap_float(srow_z[i]);
            }
        }
    } else if (sizeof_hdr == 540) { // NIfTI-2
        if (f.gcount() < 540) {
            std::cerr << "Error: Invalid NIfTI-2 file (too small)\n";
            return false;
        }
        int64_t dim[8];
        std::memcpy(dim, buf + 16, 8 * sizeof(int64_t));
        if (swap_endian) for(int i=0; i<8; i++) dim[i] = swap_int64(dim[i]);
        dims[0] = static_cast<float>(dim[1]); 
        dims[1] = static_cast<float>(dim[2]); 
        dims[2] = static_cast<float>(dim[3]);

        double pixdim[8];
        std::memcpy(pixdim, buf + 80, 8 * sizeof(double));
        if (swap_endian) for(int i=0; i<8; i++) pixdim[i] = swap_double(pixdim[i]);
        qfac = (pixdim[0] == 0.0) ? 1.0f : static_cast<float>(pixdim[0]);
        dx = static_cast<float>(pixdim[1]); 
        dy = static_cast<float>(pixdim[2]); 
        dz = static_cast<float>(pixdim[3]);

        int32_t sform32, qform32;
        std::memcpy(&qform32, buf + 344, sizeof(int32_t));
        std::memcpy(&sform32, buf + 348, sizeof(int32_t));
        if (swap_endian) { qform32 = swap_int32(qform32); sform32 = swap_int32(sform32); }
        qform_code = qform32; sform_code = sform32;

        double qb, qc, qd, qox, qoy, qoz;
        std::memcpy(&qb, buf + 352, sizeof(double));
        std::memcpy(&qc, buf + 360, sizeof(double));
        std::memcpy(&qd, buf + 368, sizeof(double));
        std::memcpy(&qox, buf + 376, sizeof(double));
        std::memcpy(&qoy, buf + 384, sizeof(double));
        std::memcpy(&qoz, buf + 392, sizeof(double));
        if (swap_endian) {
            qb = swap_double(qb); qc = swap_double(qc); qd = swap_double(qd);
            qox = swap_double(qox); qoy = swap_double(qoy); qoz = swap_double(qoz);
        }
        b = static_cast<float>(qb); c = static_cast<float>(qc); d = static_cast<float>(qd);
        qoffset_x = static_cast<float>(qox); qoffset_y = static_cast<float>(qoy); qoffset_z = static_cast<float>(qoz);
        
        double sx[4], sy[4], sz[4];
        std::memcpy(sx, buf + 400, 4 * sizeof(double));
        std::memcpy(sy, buf + 432, 4 * sizeof(double));
        std::memcpy(sz, buf + 464, 4 * sizeof(double));
        if (swap_endian) {
            for(int i=0; i<4; i++) {
                sx[i] = swap_double(sx[i]);
                sy[i] = swap_double(sy[i]);
                sz[i] = swap_double(sz[i]);
            }
        }
        for(int i=0; i<4; i++) {
            srow_x[i] = static_cast<float>(sx[i]);
            srow_y[i] = static_cast<float>(sy[i]);
            srow_z[i] = static_cast<float>(sz[i]);
        }
    } else {
        std::cerr << "Error: Unrecognized NIfTI file\n";
        return false;
    }

    float v2r[4][4];
    if (sform_code > 0) {
        for(int i=0; i<4; i++) {
            v2r[0][i] = srow_x[i];
            v2r[1][i] = srow_y[i];
            v2r[2][i] = srow_z[i];
        }
        v2r[3][0] = 0; v2r[3][1] = 0; v2r[3][2] = 0; v2r[3][3] = 1;
    } else if (qform_code > 0) {
        float b2 = b*b;
        float c2 = c*c;
        float d2 = d*d;
        float a = std::sqrt(std::max(0.0f, 1.0f - b2 - c2 - d2));
        
        float R[3][3];
        R[0][0] = a*a + b*b - c*c - d*d;
        R[0][1] = 2.0f * (b*c - a*d);
        R[0][2] = 2.0f * (b*d + a*c);
        
        R[1][0] = 2.0f * (b*c + a*d);
        R[1][1] = a*a + c*c - b*b - d*d;
        R[1][2] = 2.0f * (c*d - a*b);
        
        R[2][0] = 2.0f * (b*d - a*c);
        R[2][1] = 2.0f * (c*d + a*b);
        R[2][2] = a*a + d*d - c*c - b*b;
        
        v2r[0][0] = R[0][0] * dx; v2r[0][1] = R[0][1] * dy; v2r[0][2] = R[0][2] * qfac * dz; v2r[0][3] = qoffset_x;
        v2r[1][0] = R[1][0] * dx; v2r[1][1] = R[1][1] * dy; v2r[1][2] = R[1][2] * qfac * dz; v2r[1][3] = qoffset_y;
        v2r[2][0] = R[2][0] * dx; v2r[2][1] = R[2][1] * dy; v2r[2][2] = R[2][2] * qfac * dz; v2r[2][3] = qoffset_z;
        v2r[3][0] = 0; v2r[3][1] = 0; v2r[3][2] = 0; v2r[3][3] = 1;
    } else {
        std::cerr << "Error: NIfTI file has no valid spatial transform\n";
        return false;
    }

    out_header = json11::Json::object {
        { "DIMENSIONS", json11::Json::array { dims[0], dims[1], dims[2] } },
        { "VOXEL_TO_RASMM", json11::Json::array {
            json11::Json::array { v2r[0][0], v2r[0][1], v2r[0][2], v2r[0][3] },
            json11::Json::array { v2r[1][0], v2r[1][1], v2r[1][2], v2r[1][3] },
            json11::Json::array { v2r[2][0], v2r[2][1], v2r[2][2], v2r[2][3] },
            json11::Json::array { v2r[3][0], v2r[3][1], v2r[3][2], v2r[3][3] }
        } }
    };
    return true;
}

bool save_trx(const Tractogram &tr, const std::string &out_path, const std::string &ref_nifti_path) {
    try {
        json11::Json header_to_use = tr.header;
        if (header_to_use.is_null() || !header_to_use["VOXEL_TO_RASMM"].is_array() || header_to_use["VOXEL_TO_RASMM"].array_items().empty()) {
            if (ref_nifti_path.empty()) {
                std::cerr << "Error: TCK/VTK -> TRX requires a reference NIfTI file\n";
                return false;
            }
            json11::Json ref_hdr;
            if (!load_nifti_header(ref_nifti_path, ref_hdr)) return false;
            auto obj = header_to_use.is_object() ? header_to_use.object_items() : std::map<std::string, json11::Json>();
            obj["VOXEL_TO_RASMM"] = ref_hdr["VOXEL_TO_RASMM"];
            obj["DIMENSIONS"] = ref_hdr["DIMENSIONS"];
            header_to_use = obj;
        }

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
        trx.header = header_to_use;
        
        trx.save(out_path, trx::TrxCompression::None);
        trx.close();
        
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error saving TRX file: " << e.what() << std::endl;
        return false;
    }
}

// Simple 4x4 matrix inversion helper for save_trk
bool invert_matrix4x4(const float m[4][4], float invOut[4][4]) {
    float inv[16], det;
    float m_1d[16];
    for(int i=0; i<4; i++) for(int j=0; j<4; j++) m_1d[i*4+j] = m[i][j];

    inv[0] = m_1d[5]  * m_1d[10] * m_1d[15] - m_1d[5]  * m_1d[11] * m_1d[14] - m_1d[9]  * m_1d[6]  * m_1d[15] + m_1d[9]  * m_1d[7]  * m_1d[14] + m_1d[13] * m_1d[6]  * m_1d[11] - m_1d[13] * m_1d[7]  * m_1d[10];
    inv[4] = -m_1d[4]  * m_1d[10] * m_1d[15] + m_1d[4]  * m_1d[11] * m_1d[14] + m_1d[8]  * m_1d[6]  * m_1d[15] - m_1d[8]  * m_1d[7]  * m_1d[14] - m_1d[12] * m_1d[6]  * m_1d[11] + m_1d[12] * m_1d[7]  * m_1d[10];
    inv[8] = m_1d[4]  * m_1d[9] * m_1d[15] - m_1d[4]  * m_1d[11] * m_1d[13] - m_1d[8]  * m_1d[5] * m_1d[15] + m_1d[8]  * m_1d[7] * m_1d[13] + m_1d[12] * m_1d[5] * m_1d[11] - m_1d[12] * m_1d[7] * m_1d[9];
    inv[12] = -m_1d[4]  * m_1d[9] * m_1d[14] + m_1d[4]  * m_1d[10] * m_1d[13] + m_1d[8]  * m_1d[5] * m_1d[14] - m_1d[8]  * m_1d[6] * m_1d[13] - m_1d[12] * m_1d[5] * m_1d[10] + m_1d[12] * m_1d[6] * m_1d[9];
    inv[1] = -m_1d[1]  * m_1d[10] * m_1d[15] + m_1d[1]  * m_1d[11] * m_1d[14] + m_1d[9]  * m_1d[2] * m_1d[15] - m_1d[9]  * m_1d[3] * m_1d[14] - m_1d[13] * m_1d[2] * m_1d[11] + m_1d[13] * m_1d[3] * m_1d[10];
    inv[5] = m_1d[0]  * m_1d[10] * m_1d[15] - m_1d[0]  * m_1d[11] * m_1d[14] - m_1d[8]  * m_1d[2] * m_1d[15] + m_1d[8]  * m_1d[3] * m_1d[14] + m_1d[12] * m_1d[2] * m_1d[11] - m_1d[12] * m_1d[3] * m_1d[10];
    inv[9] = -m_1d[0]  * m_1d[9] * m_1d[15] + m_1d[0]  * m_1d[11] * m_1d[13] + m_1d[8]  * m_1d[1] * m_1d[15] - m_1d[8]  * m_1d[3] * m_1d[13] - m_1d[12] * m_1d[1] * m_1d[11] + m_1d[12] * m_1d[3] * m_1d[9];
    inv[13] = m_1d[0]  * m_1d[9] * m_1d[14] - m_1d[0]  * m_1d[10] * m_1d[13] - m_1d[8]  * m_1d[1] * m_1d[14] + m_1d[8]  * m_1d[2] * m_1d[13] + m_1d[12] * m_1d[1] * m_1d[10] - m_1d[12] * m_1d[2] * m_1d[9];
    inv[2] = m_1d[1]  * m_1d[6] * m_1d[15] - m_1d[1]  * m_1d[7] * m_1d[14] - m_1d[5]  * m_1d[2] * m_1d[15] + m_1d[5]  * m_1d[3] * m_1d[14] + m_1d[13] * m_1d[2] * m_1d[7] - m_1d[13] * m_1d[3] * m_1d[6];
    inv[6] = -m_1d[0]  * m_1d[6] * m_1d[15] + m_1d[0]  * m_1d[7] * m_1d[14] + m_1d[4]  * m_1d[2] * m_1d[15] - m_1d[4]  * m_1d[3] * m_1d[14] - m_1d[12] * m_1d[2] * m_1d[7] + m_1d[12] * m_1d[3] * m_1d[6];
    inv[10] = m_1d[0]  * m_1d[5] * m_1d[15] - m_1d[0]  * m_1d[7] * m_1d[13] - m_1d[4]  * m_1d[1] * m_1d[15] + m_1d[4]  * m_1d[3] * m_1d[13] + m_1d[12] * m_1d[1] * m_1d[7] - m_1d[12] * m_1d[3] * m_1d[5];
    inv[14] = -m_1d[0]  * m_1d[5] * m_1d[14] + m_1d[0]  * m_1d[6] * m_1d[13] + m_1d[4]  * m_1d[1] * m_1d[14] - m_1d[4]  * m_1d[2] * m_1d[13] - m_1d[12] * m_1d[1] * m_1d[6] + m_1d[12] * m_1d[2] * m_1d[5];
    inv[3] = -m_1d[1] * m_1d[6] * m_1d[11] + m_1d[1] * m_1d[7] * m_1d[10] + m_1d[5] * m_1d[2] * m_1d[11] - m_1d[5] * m_1d[3] * m_1d[10] - m_1d[9] * m_1d[2] * m_1d[7] + m_1d[9] * m_1d[3] * m_1d[6];
    inv[7] = m_1d[0] * m_1d[6] * m_1d[11] - m_1d[0] * m_1d[7] * m_1d[10] - m_1d[4] * m_1d[2] * m_1d[11] + m_1d[4] * m_1d[3] * m_1d[10] + m_1d[8] * m_1d[2] * m_1d[7] - m_1d[8] * m_1d[3] * m_1d[6];
    inv[11] = -m_1d[0] * m_1d[5] * m_1d[11] + m_1d[0] * m_1d[7] * m_1d[9] + m_1d[4] * m_1d[1] * m_1d[11] - m_1d[4] * m_1d[3] * m_1d[9] - m_1d[8] * m_1d[1] * m_1d[7] + m_1d[8] * m_1d[3] * m_1d[5];
    inv[15] = m_1d[0] * m_1d[5] * m_1d[10] - m_1d[0] * m_1d[6] * m_1d[9] - m_1d[4] * m_1d[1] * m_1d[10] + m_1d[4] * m_1d[2] * m_1d[9] + m_1d[8] * m_1d[1] * m_1d[6] - m_1d[8] * m_1d[2] * m_1d[5];

    det = m_1d[0] * inv[0] + m_1d[1] * inv[4] + m_1d[2] * inv[8] + m_1d[3] * inv[12];
    if (det == 0) return false;
    det = 1.0f / det;
    for (int i = 0; i < 16; i++) {
        invOut[i/4][i%4] = inv[i] * det;
    }
    return true;
}

bool save_trk(const Tractogram &tr, const std::string &out_path, const std::string &original_filename, const std::string &ref_nifti_path) {
    std::ofstream f(out_path, std::ios::binary);
    if (!f.is_open()) return false;

    json11::Json header_to_use = tr.header;
    if (header_to_use.is_null() || !header_to_use["VOXEL_TO_RASMM"].is_array() || header_to_use["VOXEL_TO_RASMM"].array_items().empty()) {
        if (ref_nifti_path.empty()) {
            std::cerr << "Error: TCK/VTK -> TRK requires a reference NIfTI file\n";
            return false;
        }
        json11::Json ref_hdr;
        if (!load_nifti_header(ref_nifti_path, ref_hdr)) return false;
        auto obj = header_to_use.is_object() ? header_to_use.object_items() : std::map<std::string, json11::Json>();
        obj["VOXEL_TO_RASMM"] = ref_hdr["VOXEL_TO_RASMM"];
        obj["DIMENSIONS"] = ref_hdr["DIMENSIONS"];
        header_to_use = obj;
    }

    TrkHeader header;
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic_number, "TRACK", 5);

    // Initialize vox_to_rasmm to identity
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            header.voxel_to_rasmm[r][c] = (r == c) ? 1.0f : 0.0f;

    // Attempt to extract from JSON header
    if (header_to_use["DIMENSIONS"].is_array()) {
        auto dims = header_to_use["DIMENSIONS"].array_items();
        if (dims.size() >= 3) {
            header.dimensions[0] = static_cast<int16_t>(dims[0].number_value());
            header.dimensions[1] = static_cast<int16_t>(dims[1].number_value());
            header.dimensions[2] = static_cast<int16_t>(dims[2].number_value());
        }
    }
    if (header_to_use["VOXEL_TO_RASMM"].is_array()) {
        auto rows = header_to_use["VOXEL_TO_RASMM"].array_items();
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

    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            mat(r, c) = header.voxel_to_rasmm[r][c];
    Eigen::Matrix4f inv_mat = mat.inverse();

    float vx = header.voxel_sizes[0] > 0 ? header.voxel_sizes[0] : 1.0f;
    float vy = header.voxel_sizes[1] > 0 ? header.voxel_sizes[1] : 1.0f;
    float vz = header.voxel_sizes[2] > 0 ? header.voxel_sizes[2] : 1.0f;

    size_t num_streamlines = tr.offsets.size() - 1;
    std::vector<char> chunk;
    chunk.reserve(4 * 1024 * 1024);

    for (size_t i = 0; i < num_streamlines; ++i) {
        size_t start = tr.offsets[i];
        size_t end = tr.offsets[i+1];
        int32_t n_pts = static_cast<int32_t>(end - start);

        const char* p_n_pts = reinterpret_cast<const char*>(&n_pts);
        chunk.insert(chunk.end(), p_n_pts, p_n_pts + 4);

        for (size_t j = start; j < end; ++j) {
            Eigen::Vector4f p_ras(tr.pts[j*3], tr.pts[j*3 + 1], tr.pts[j*3 + 2], 1.0f);
            Eigen::Vector4f p_center = inv_mat * p_ras;

            float x = (p_center.x() + 0.5f) * vx;
            float y = (p_center.y() + 0.5f) * vy;
            float z = (p_center.z() + 0.5f) * vz;

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
