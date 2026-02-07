#include <trx/nifti_io.h>

#include <Eigen/LU>
#include <Eigen/SVD>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include <zlib.h>

namespace trx {
namespace detail {

#pragma pack(push, 1)
struct NiftiHeader {
  std::int32_t sizeof_hdr;
  char data_type[10];
  char db_name[18];
  std::int32_t extents;
  std::int16_t session_error;
  char regular;
  char dim_info;
  std::int16_t dim[8];
  float intent_p1;
  float intent_p2;
  float intent_p3;
  std::int16_t intent_code;
  std::int16_t datatype;
  std::int16_t bitpix;
  std::int16_t slice_start;
  float pixdim[8];
  float vox_offset;
  float scl_slope;
  float scl_inter;
  std::int16_t slice_end;
  char slice_code;
  char xyzt_units;
  float cal_max;
  float cal_min;
  float slice_duration;
  float toffset;
  std::int32_t glmax;
  std::int32_t glmin;
  char descrip[80];
  char aux_file[24];
  std::int16_t qform_code;
  std::int16_t sform_code;
  float quatern_b;
  float quatern_c;
  float quatern_d;
  float qoffset_x;
  float qoffset_y;
  float qoffset_z;
  float srow_x[4];
  float srow_y[4];
  float srow_z[4];
  char intent_name[16];
  char magic[4];
};
#pragma pack(pop)

static_assert(sizeof(NiftiHeader) == 348, "NIfTI-1 header must be 348 bytes");

bool has_gz_extension(const std::string &path) {
  const std::string suffix = ".gz";
  if (path.size() < suffix.size()) {
    return false;
  }
  return path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void swap_2(std::int16_t &value) {
  const uint16_t u = static_cast<uint16_t>(value);
  const uint16_t swapped = static_cast<uint16_t>((u >> 8) | (u << 8));
  value = static_cast<std::int16_t>(swapped);
}

void swap_4(std::int32_t &value) {
  const uint32_t u = static_cast<uint32_t>(value);
  const uint32_t swapped =
      ((u >> 24) & 0x000000FFu) |
      ((u >> 8) & 0x0000FF00u) |
      ((u << 8) & 0x00FF0000u) |
      ((u << 24) & 0xFF000000u);
  value = static_cast<std::int32_t>(swapped);
}

void swap_4(float &value) {
  uint32_t u = 0;
  std::memcpy(&u, &value, sizeof(u));
  const uint32_t swapped =
      ((u >> 24) & 0x000000FFu) |
      ((u >> 8) & 0x0000FF00u) |
      ((u << 8) & 0x00FF0000u) |
      ((u << 24) & 0xFF000000u);
  std::memcpy(&value, &swapped, sizeof(value));
}

void swap_nifti_header(NiftiHeader &hdr) {
  swap_4(hdr.sizeof_hdr);
  swap_4(hdr.extents);
  swap_2(hdr.session_error);
  for (std::int16_t &dim : hdr.dim) {
    swap_2(dim);
  }
  swap_4(hdr.intent_p1);
  swap_4(hdr.intent_p2);
  swap_4(hdr.intent_p3);
  swap_2(hdr.intent_code);
  swap_2(hdr.datatype);
  swap_2(hdr.bitpix);
  swap_2(hdr.slice_start);
  for (float &pixdim : hdr.pixdim) {
    swap_4(pixdim);
  }
  swap_4(hdr.vox_offset);
  swap_4(hdr.scl_slope);
  swap_4(hdr.scl_inter);
  swap_2(hdr.slice_end);
  swap_4(hdr.cal_max);
  swap_4(hdr.cal_min);
  swap_4(hdr.slice_duration);
  swap_4(hdr.toffset);
  swap_4(hdr.glmax);
  swap_4(hdr.glmin);
  swap_2(hdr.qform_code);
  swap_2(hdr.sform_code);
  swap_4(hdr.quatern_b);
  swap_4(hdr.quatern_c);
  swap_4(hdr.quatern_d);
  swap_4(hdr.qoffset_x);
  swap_4(hdr.qoffset_y);
  swap_4(hdr.qoffset_z);
  for (float &value : hdr.srow_x) {
    swap_4(value);
  }
  for (float &value : hdr.srow_y) {
    swap_4(value);
  }
  for (float &value : hdr.srow_z) {
    swap_4(value);
  }
}

void read_bytes_gz(const std::string &path, void *buffer, size_t length) {
  gzFile file = gzopen(path.c_str(), "rb");
  if (!file) {
    throw std::runtime_error("Failed to open gzip NIfTI header: " + path);
  }
  size_t total = 0;
  while (total < length) {
    const int read_now = gzread(file, static_cast<char *>(buffer) + total,
                                static_cast<unsigned int>(length - total));
    if (read_now <= 0) {
      gzclose(file);
      throw std::runtime_error("Failed to read gzip NIfTI header: " + path);
    }
    total += static_cast<size_t>(read_now);
  }
  gzclose(file);
}

void read_bytes_raw(const std::string &path, void *buffer, size_t length) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open NIfTI header: " + path);
  }
  in.read(static_cast<char *>(buffer), static_cast<std::streamsize>(length));
  if (in.gcount() != static_cast<std::streamsize>(length)) {
    throw std::runtime_error("Failed to read NIfTI header: " + path);
  }
}

NiftiHeader read_header(const std::string &path) {
  NiftiHeader hdr{};
  if (has_gz_extension(path)) {
    read_bytes_gz(path, &hdr, sizeof(NiftiHeader));
  } else {
    read_bytes_raw(path, &hdr, sizeof(NiftiHeader));
  }

  if (hdr.sizeof_hdr != 348) {
    swap_nifti_header(hdr);
    if (hdr.sizeof_hdr != 348) {
      throw std::runtime_error("Invalid NIfTI header size for: " + path);
    }
  }

  return hdr;
}

Eigen::Matrix4f quatern_to_mat44(float qb,
                                 float qc,
                                 float qd,
                                 float qx,
                                 float qy,
                                 float qz,
                                 float dx,
                                 float dy,
                                 float dz,
                                 float qfac) {
  Eigen::Matrix4f R = Eigen::Matrix4f::Identity();

  double a = 1.0 - (qb * qb + qc * qc + qd * qd);
  double b = qb;
  double c = qc;
  double d = qd;

  if (a < 1.e-7) {
    a = 1.0 / std::sqrt(b * b + c * c + d * d);
    b *= a;
    c *= a;
    d *= a;
    a = 0.0;
  } else {
    a = std::sqrt(a);
  }

  const double xd = (dx > 0.0f) ? dx : 1.0;
  const double yd = (dy > 0.0f) ? dy : 1.0;
  double zd = (dz > 0.0f) ? dz : 1.0;
  if (qfac < 0.0f) {
    zd = -zd;
  }

  R(0, 0) = static_cast<float>((a * a + b * b - c * c - d * d) * xd);
  R(0, 1) = static_cast<float>(2.0 * (b * c - a * d) * yd);
  R(0, 2) = static_cast<float>(2.0 * (b * d + a * c) * zd);
  R(1, 0) = static_cast<float>(2.0 * (b * c + a * d) * xd);
  R(1, 1) = static_cast<float>((a * a + c * c - b * b - d * d) * yd);
  R(1, 2) = static_cast<float>(2.0 * (c * d - a * b) * zd);
  R(2, 0) = static_cast<float>(2.0 * (b * d - a * c) * xd);
  R(2, 1) = static_cast<float>(2.0 * (c * d + a * b) * yd);
  R(2, 2) = static_cast<float>((a * a + d * d - c * c - b * b) * zd);

  R(0, 3) = qx;
  R(1, 3) = qy;
  R(2, 3) = qz;
  return R;
}

Eigen::Matrix4f sform_to_qform_matrix(const NiftiHeader &hdr) {
  Eigen::Matrix3f M;
  M << hdr.srow_x[0], hdr.srow_x[1], hdr.srow_x[2],
      hdr.srow_y[0], hdr.srow_y[1], hdr.srow_y[2],
      hdr.srow_z[0], hdr.srow_z[1], hdr.srow_z[2];

  float dx = M.col(0).norm();
  float dy = M.col(1).norm();
  float dz = M.col(2).norm();

  if (dx == 0.0f) {
    dx = 1.0f;
    M.col(0) = Eigen::Vector3f::UnitX();
  }
  if (dy == 0.0f) {
    dy = 1.0f;
    M.col(1) = Eigen::Vector3f::UnitY();
  }
  if (dz == 0.0f) {
    dz = 1.0f;
    M.col(2) = Eigen::Vector3f::UnitZ();
  }

  Eigen::Matrix3f R = M;
  R.col(0) /= dx;
  R.col(1) /= dy;
  R.col(2) /= dz;

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f R_orth = svd.matrixU() * svd.matrixV().transpose();

  if (R_orth.determinant() < 0.0f) {
    R_orth.col(2) *= -1.0f;
    dz = -dz;
  }

  Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
  out.block<3, 3>(0, 0) = R_orth * Eigen::DiagonalMatrix<float, 3>(dx, dy, dz);
  out(0, 3) = hdr.srow_x[3];
  out(1, 3) = hdr.srow_y[3];
  out(2, 3) = hdr.srow_z[3];
  return out;
}

} // namespace detail

Eigen::Matrix4f read_nifti_voxel_to_rasmm(const std::string &path) {
  const detail::NiftiHeader hdr = detail::read_header(path);

  if (hdr.qform_code > 0) {
    float qfac = hdr.pixdim[0];
    if (qfac == 0.0f) {
      qfac = 1.0f;
    } else if (qfac > 0.0f) {
      qfac = 1.0f;
    } else {
      qfac = -1.0f;
    }
    return detail::quatern_to_mat44(hdr.quatern_b,
                                    hdr.quatern_c,
                                    hdr.quatern_d,
                                    hdr.qoffset_x,
                                    hdr.qoffset_y,
                                    hdr.qoffset_z,
                                    hdr.pixdim[1],
                                    hdr.pixdim[2],
                                    hdr.pixdim[3],
                                    qfac);
  }

  if (hdr.sform_code > 0) {
    return detail::sform_to_qform_matrix(hdr);
  }

  throw std::runtime_error("NIfTI header has no qform or sform: " + path);
}

} // namespace trx
