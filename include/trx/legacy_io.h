#ifndef TRX_LEGACY_IO_H
#define TRX_LEGACY_IO_H

#include <string>
#include <vector>
#include <cstdint>
#include <trx/trx.h>

namespace trx {
namespace legacy {

struct Tractogram {
    std::vector<float> pts;
    std::vector<uint64_t> offsets;
    json11::Json header;
    std::shared_ptr<trx::AnyTrxFile> original_trx;
};

#pragma pack(push, 1)
struct TrkHeader {
    char magic_number[6];
    int16_t dimensions[3];
    float voxel_sizes[3];
    float origin[3];
    int16_t nb_scalars_per_point;
    char scalar_name[10][20];
    int16_t nb_properties_per_streamline;
    char property_name[10][20];
    float voxel_to_rasmm[4][4];
    char reserved[444];
    char voxel_order[4];
    char pad2[4];
    float image_orientation_patient[6];
    char pad1[2];
    char invert_x;
    char invert_y;
    char invert_z;
    char swap_xy;
    char swap_yz;
    char swap_zx;
    int32_t nb_streamlines;
    int32_t version;
    int32_t hdr_size;
};
#pragma pack(pop)

bool load_trx(const std::string &filename, Tractogram &tr);
bool load_trk(const std::string &filename, Tractogram &tr);
bool load_tck(const std::string &filename, Tractogram &tr);
bool load_vtk(const std::string &filename, Tractogram &tr);

bool save_trx(const Tractogram &tr, const std::string &out_path);
bool save_trk(const Tractogram &tr, const std::string &out_path, const std::string &original_filename = "");
bool save_tck(const Tractogram &tr, const std::string &out_path);
bool save_vtk(const Tractogram &tr, const std::string &out_path);

} // namespace legacy
} // namespace trx

#endif // TRX_LEGACY_IO_H
