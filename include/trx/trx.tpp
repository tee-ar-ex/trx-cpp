// Taken from: https://stackoverflow.com/a/25389481
#ifndef TRX_H
#define TRX_TPP_STANDALONE
#define TRX_TPP_OPEN_NAMESPACE
#include <trx/trx.h>
#undef TRX_TPP_STANDALONE
namespace trx {
#endif
using Eigen::Dynamic;
using Eigen::half;
using Eigen::Index;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;
template <class Matrix> void write_binary(const std::string &filename, const Matrix &matrix) {
  std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
  typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
  // out.write((char *)(&rows), sizeof(typename Matrix::Index));
  // out.write((char *)(&cols), sizeof(typename Matrix::Index));
  const auto *data = reinterpret_cast<const char *>(matrix.data()); // check_syntax off
  out.write(data, rows * cols * sizeof(typename Matrix::Scalar));
  out.close();
}
template <class Matrix> void read_binary(const std::string &filename, Matrix &matrix) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  typename Matrix::Index rows = 0, cols = 0;
  auto *rows_ptr = reinterpret_cast<char *>(&rows); // check_syntax off
  auto *cols_ptr = reinterpret_cast<char *>(&cols); // check_syntax off
  in.read(rows_ptr, sizeof(typename Matrix::Index));
  in.read(cols_ptr, sizeof(typename Matrix::Index));
  matrix.resize(rows, cols);
  auto *matrix_ptr = reinterpret_cast<char *>(matrix.data()); // check_syntax off
  in.read(matrix_ptr, rows * cols * sizeof(typename Matrix::Scalar));
  in.close();
}

template <typename DT>
void ediff1d(Matrix<DT, Dynamic, 1> &lengths, Matrix<DT, Dynamic, Dynamic> &tmp, uint32_t to_end) {
  Map<Matrix<uint32_t, 1, Dynamic>> v(tmp.data(), tmp.size());
  lengths.resize(v.size(), 1);

  // TODO: figure out if there's a built in way to manage this
  for (int i = 0; i < v.size() - 1; i++) {
    lengths(i) = v(i + 1) - v(i);
  }
  lengths(v.size() - 1) = to_end;
}

template <typename DT>
// Caveat: if filename has an extension, it will be replaced by the generated dtype extension.
std::string _generate_filename_from_data(const Eigen::MatrixBase<DT> &arr, std::string filename) {

  std::string base, ext;

  base = filename; // get_base(SEPARATOR, filename);
  ext = get_ext(filename);

  if (ext.size() != 0) {
    base = base.substr(0, base.length() - ext.length() - 1);
  }

  std::string dt = dtype_from_scalar<typename DT::Scalar>();

  Eigen::Index n_cols = arr.cols();

  std::string new_filename;
  if (n_cols == 1) {
    new_filename = base + "." + dt;
  } else {
    new_filename = base + "." + std::to_string(static_cast<long long>(n_cols)) + "." + dt;
  }

  return new_filename;
}

template <typename DT>
TrxFile<DT>::TrxFile(int nb_vertices, int nb_streamlines, const TrxFile<DT> *init_as, std::string reference) {
  std::vector<std::vector<float>> affine(4);
  std::vector<uint16_t> dimensions(3);

  // TODO: check if there's a more efficient way to do this with Eigen
  if (init_as != nullptr) {
    for (int i = 0; i < 4; i++) {
      affine[i] = {0, 0, 0, 0};
      for (int j = 0; j < 4; j++) {
        affine[i][j] = static_cast<float>(init_as->header["VOXEL_TO_RASMM"][i][j].number_value());
      }
    }

    for (int i = 0; i < 3; i++) {
      dimensions[i] = static_cast<uint16_t>(init_as->header["DIMENSIONS"][i].int_value());
    }
  }
  // TODO: add else if for get_reference_info
  else {
    // identity matrix
    for (int i = 0; i < 4; i++) {
      affine[i] = {0, 0, 0, 0};
      affine[i][i] = 1;
    }
    dimensions = {1, 1, 1};
  }

  if (nb_vertices == 0 && nb_streamlines == 0) {
    if (init_as != nullptr) {
      // raise error here
      throw std::invalid_argument("Can't us init_as without declaring nb_vertices and nb_streamlines");
    }

    // will remove as completely unecessary. using as placeholders
    this->header = {};
    this->streamlines.reset();

    // TODO: maybe create a matrix to map to of specified DT. Do we need this??
    // set default datatype to half
    // default data is null so will not set data. User will need configure desired datatype
    // this->streamlines = ArraySequence<half>();
    this->_uncompressed_folder_handle = "";

    nb_vertices = 0;
    nb_streamlines = 0;
  } else if (nb_vertices > 0 && nb_streamlines > 0) {
    auto trx = _initialize_empty_trx<DT>(nb_streamlines, nb_vertices, init_as);
    this->streamlines = std::move(trx->streamlines);
    this->groups = std::move(trx->groups);
    this->data_per_streamline = std::move(trx->data_per_streamline);
    this->data_per_vertex = std::move(trx->data_per_vertex);
    this->data_per_group = std::move(trx->data_per_group);
    this->_uncompressed_folder_handle = std::move(trx->_uncompressed_folder_handle);
    this->_owns_uncompressed_folder = trx->_owns_uncompressed_folder;
    this->_copy_safe = trx->_copy_safe;
    trx->_owns_uncompressed_folder = false;
    trx->_uncompressed_folder_handle.clear();
  } else {
    throw std::invalid_argument("You must declare both NB_VERTICES AND NB_STREAMLINES");
  }

  json::object header_obj;
  header_obj["VOXEL_TO_RASMM"] = affine;
  header_obj["DIMENSIONS"] = dimensions;
  header_obj["NB_VERTICES"] = nb_vertices;
  header_obj["NB_STREAMLINES"] = nb_streamlines;
  this->header = json(header_obj);

  this->_copy_safe = true;
}

template <typename DT>
std::unique_ptr<TrxFile<DT>> _initialize_empty_trx(int nb_streamlines, int nb_vertices, const TrxFile<DT> *init_as) {
  auto trx = std::make_unique<TrxFile<DT>>();

  std::string tmp_dir = make_temp_dir("trx");

  json header = json::object();
  if (init_as != nullptr) {
    header = init_as->header;
  }
  header = _json_set(header, "NB_VERTICES", nb_vertices);
  header = _json_set(header, "NB_STREAMLINES", nb_streamlines);

  std::string positions_dtype;
  std::string offsets_dtype;
  std::string lengths_dtype;

  if (init_as != nullptr) {
    header = _json_set(header, "VOXEL_TO_RASMM", init_as->header["VOXEL_TO_RASMM"]);
    header = _json_set(header, "DIMENSIONS", init_as->header["DIMENSIONS"]);
    positions_dtype = dtype_from_scalar<DT>();
    offsets_dtype = dtype_from_scalar<uint64_t>();
    lengths_dtype = dtype_from_scalar<uint32_t>();
  } else {
    positions_dtype = dtype_from_scalar<DT>();
    offsets_dtype = dtype_from_scalar<uint64_t>();
    lengths_dtype = dtype_from_scalar<uint32_t>();
  }
  std::string positions_filename(tmp_dir);
  positions_filename += "/positions.3." + positions_dtype;

  std::tuple<int, int> shape = std::make_tuple(nb_vertices, 3);

  trx->streamlines = std::make_unique<ArraySequence<DT>>();
  trx->streamlines->mmap_pos = trx::_create_memmap(positions_filename, shape, "w+", positions_dtype);

  // TODO: find a better way to get the dtype than using all these switch cases.
  if (positions_dtype.compare("float16") == 0) {
    new (&(trx->streamlines->_data)) Map<Matrix<half, Dynamic, Dynamic>>(
        reinterpret_cast<half *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
  } else if (positions_dtype.compare("float32") == 0) {
    new (&(trx->streamlines->_data)) Map<Matrix<float, Dynamic, Dynamic>>(
        reinterpret_cast<float *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
  } else {
    new (&(trx->streamlines->_data)) Map<Matrix<double, Dynamic, Dynamic>>(
        reinterpret_cast<double *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
  }

  std::string offsets_filename(tmp_dir);
  offsets_filename += "/offsets." + offsets_dtype;

  std::tuple<int, int> shape_off = std::make_tuple(nb_streamlines + 1, 1);

  trx->streamlines->mmap_off = trx::_create_memmap(offsets_filename, shape_off, "w+", offsets_dtype);
  new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(
      reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape_off), std::get<1>(shape_off));

  trx->streamlines->_lengths.resize(nb_streamlines);
  trx->streamlines->_lengths.setZero();

  if (init_as != nullptr) {
    std::string dpv_dirname;
    std::string dps_dirname;
    if (init_as->data_per_vertex.size() > 0) {
      dpv_dirname = tmp_dir + "/dpv/";
      std::error_code ec;
      trx::fs::create_directories(dpv_dirname, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + dpv_dirname);
      }
    }
    if (init_as->data_per_streamline.size() > 0) {
      dps_dirname = tmp_dir + "/dps/";
      std::error_code ec;
      trx::fs::create_directories(dps_dirname, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + dps_dirname);
      }
    }

    for (auto const &x : init_as->data_per_vertex) {
      int rows, cols;
      std::string dpv_dtype = dtype_from_scalar<DT>();
      Map<Matrix<DT, Dynamic, Dynamic, RowMajor>> tmp_as = init_as->data_per_vertex.find(x.first)->second->_data;

      std::string dpv_filename;
      if (tmp_as.rows() == 1) {
        dpv_filename = dpv_dirname + x.first + "." + dpv_dtype;
        rows = nb_vertices;
        cols = 1;
      } else {
        rows = nb_vertices;
        cols = tmp_as.cols();

        dpv_filename = dpv_dirname + x.first + "." + std::to_string(cols) + "." + dpv_dtype;
      }

      std::tuple<int, int> dpv_shape = std::make_tuple(rows, cols);
      trx->data_per_vertex[x.first] = std::make_unique<ArraySequence<DT>>();
      trx->data_per_vertex[x.first]->mmap_pos = trx::_create_memmap(dpv_filename, dpv_shape, "w+", dpv_dtype);
      if (dpv_dtype.compare("float16") == 0) {
        new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<half, Dynamic, Dynamic>>(
            reinterpret_cast<half *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
      } else if (dpv_dtype.compare("float32") == 0) {
        new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<float, Dynamic, Dynamic>>(
            reinterpret_cast<float *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
      } else {
        new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<double, Dynamic, Dynamic>>(
            reinterpret_cast<double *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
      }

      new (&(trx->data_per_vertex[x.first]->_offsets))
          Map<Matrix<uint64_t, Dynamic, Dynamic>>(trx->streamlines->_offsets.data(),
                                                  int(trx->streamlines->_offsets.rows()),
                                                  int(trx->streamlines->_offsets.cols()));
      trx->data_per_vertex[x.first]->_lengths = trx->streamlines->_lengths;
    }

    for (auto const &x : init_as->data_per_streamline) {
      std::string dps_dtype = dtype_from_scalar<DT>();
      int rows, cols;
      Map<Matrix<DT, Dynamic, Dynamic>> tmp_as = init_as->data_per_streamline.find(x.first)->second->_matrix;

      std::string dps_filename;

      if (tmp_as.rows() == 1) {
        dps_filename = dps_dirname + x.first + "." + dps_dtype;
        rows = nb_streamlines;
      } else {
        cols = tmp_as.cols();
        rows = nb_streamlines;

        dps_filename = dps_dirname + x.first + "." + std::to_string(cols) + "." + dps_dtype;
      }

      std::tuple<int, int> dps_shape = std::make_tuple(rows, cols);
      trx->data_per_streamline[x.first] = std::make_unique<trx::MMappedMatrix<DT>>();
      trx->data_per_streamline[x.first]->mmap =
          trx::_create_memmap(dps_filename, dps_shape, std::string("w+"), dps_dtype);

      if (dps_dtype.compare("float16") == 0) {
        new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(
            reinterpret_cast<half *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
      } else if (dps_dtype.compare("float32") == 0) {
        new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(
            reinterpret_cast<float *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
      } else {
        new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(
            reinterpret_cast<double *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
      }
    }
  }

  trx->header = header;
  trx->_uncompressed_folder_handle = tmp_dir;
  trx->_owns_uncompressed_folder = true;

  return trx;
}

template <typename DT>
std::unique_ptr<TrxFile<DT>>
TrxFile<DT>::_create_trx_from_pointer(json header,
                                      std::map<std::string, std::tuple<long long, long long>> dict_pointer_size,
                                      std::string root_zip,
                                      std::string root) {
  auto trx = std::make_unique<trx::TrxFile<DT>>();
  trx->header = header;
  trx->streamlines = std::make_unique<ArraySequence<DT>>();

  std::string filename;

  // TODO: Fix this hack of iterating through dictionary in reverse to get main files read first
  for (auto x = dict_pointer_size.rbegin(); x != dict_pointer_size.rend(); ++x) {
    std::string elem_filename = x->first;

    if (root_zip.size() > 0) {
      filename = root_zip;
    } else {
      filename = elem_filename;
    }

    trx::fs::path elem_path(elem_filename);
    trx::fs::path folder_path = elem_path.parent_path();
    std::string folder;
    if (!root.empty()) {
      trx::fs::path rel_path = elem_path.lexically_relative(trx::fs::path(root));
      std::string rel_str = rel_path.string();
      if (!rel_str.empty() && rel_str.rfind("..", 0) != 0) {
        folder = rel_path.parent_path().string();
      } else {
        folder = folder_path.string();
      }
    } else {
      folder = folder_path.string();
    }
    if (folder == ".") {
      folder.clear();
    }

    // _split_ext_with_dimensionality
    std::tuple<std::string, int, std::string> base_tuple = trx::detail::_split_ext_with_dimensionality(elem_filename);
    std::string base(std::get<0>(base_tuple));
    int dim = std::get<1>(base_tuple);
    std::string ext(std::get<2>(base_tuple));

    long long mem_adress = std::get<0>(x->second);
    long long size = std::get<1>(x->second);

    if (base.compare("positions") == 0 && (folder.compare("") == 0 || folder.compare(".") == 0)) {
      const auto nb_vertices = static_cast<int64_t>(trx->header["NB_VERTICES"].int_value());
      const auto expected = nb_vertices * 3;
      if (size != expected || dim != 3) {
        throw std::invalid_argument("Wrong data size/dimensionality: size=" + std::to_string(size) +
                                    " expected=" + std::to_string(expected) + " dim=" + std::to_string(dim) +
                                    " filename=" + elem_filename);
      }

      std::tuple<int, int> shape = std::make_tuple(static_cast<int>(trx->header["NB_VERTICES"].int_value()), 3);
      trx->streamlines->mmap_pos =
          trx::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

      // TODO: find a better way to get the dtype than using all these switch cases. Also
      // refactor into function as per specifications, positions can only be floats
      if (ext.compare("float16") == 0) {
        new (&(trx->streamlines->_data)) Map<Matrix<half, Dynamic, Dynamic>>(
            reinterpret_cast<half *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
      } else if (ext.compare("float32") == 0) {
        new (&(trx->streamlines->_data)) Map<Matrix<float, Dynamic, Dynamic>>(
            reinterpret_cast<float *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
      } else {
        new (&(trx->streamlines->_data)) Map<Matrix<double, Dynamic, Dynamic>>(
            reinterpret_cast<double *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
      }
    }

    else if (base.compare("offsets") == 0 && (folder.compare("") == 0 || folder.compare(".") == 0)) {
      const auto nb_streamlines = static_cast<int64_t>(trx->header["NB_STREAMLINES"].int_value());
      const auto expected = nb_streamlines + 1;
      if (size != expected || dim != 1) {
        throw std::invalid_argument("Wrong offsets size/dimensionality: size=" + std::to_string(size) +
                                    " expected=" + std::to_string(expected) + " dim=" + std::to_string(dim) +
                                    " filename=" + elem_filename);
      }

      const int nb_str = static_cast<int>(trx->header["NB_STREAMLINES"].int_value());
      std::tuple<int, int> shape = std::make_tuple(nb_str + 1, 1);
      trx->streamlines->mmap_off = trx::_create_memmap(filename, shape, "r+", ext, mem_adress);

      if (ext.compare("uint64") == 0) {
        new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(
            reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape), std::get<1>(shape));
      } else if (ext.compare("uint32") == 0) {
        trx->streamlines->_offsets_owned.resize(std::get<0>(shape));
        auto *src = reinterpret_cast<const uint32_t *>(trx->streamlines->mmap_off.data());
        for (int i = 0; i < std::get<0>(shape); ++i)
          trx->streamlines->_offsets_owned[static_cast<size_t>(i)] = static_cast<uint64_t>(src[i]);
        new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(
            trx->streamlines->_offsets_owned.data(), std::get<0>(shape), std::get<1>(shape));
      } else {
        throw std::invalid_argument("Unsupported offsets datatype: " + ext);
      }

      Matrix<uint64_t, Dynamic, 1> offsets = trx->streamlines->_offsets;
      trx->streamlines->_lengths =
          trx::detail::_compute_lengths(offsets, static_cast<int>(trx->header["NB_VERTICES"].int_value()));
    }

    else if (folder.compare("dps") == 0) {
      std::tuple<int, int> shape;
      trx->data_per_streamline[base] = std::make_unique<MMappedMatrix<DT>>();
      int nb_scalar = size / static_cast<int>(trx->header["NB_STREAMLINES"].int_value());

      if (size % static_cast<int>(trx->header["NB_STREAMLINES"].int_value()) != 0 || nb_scalar != dim) {

        throw std::invalid_argument("Wrong dps size/dimensionality");
      } else {
        shape = std::make_tuple(static_cast<int>(trx->header["NB_STREAMLINES"].int_value()), nb_scalar);
      }
      trx->data_per_streamline[base]->mmap = trx::_create_memmap(filename, shape, "r+", ext, mem_adress);

      if (ext.compare("float16") == 0) {
        new (&(trx->data_per_streamline[base]->_matrix))
            Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_streamline[base]->mmap.data()),
                                                std::get<0>(shape),
                                                std::get<1>(shape));
      } else if (ext.compare("float32") == 0) {
        new (&(trx->data_per_streamline[base]->_matrix))
            Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_streamline[base]->mmap.data()),
                                                 std::get<0>(shape),
                                                 std::get<1>(shape));
      } else {
        new (&(trx->data_per_streamline[base]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(
            reinterpret_cast<double *>(trx->data_per_streamline[base]->mmap.data()),
            std::get<0>(shape),
            std::get<1>(shape));
      }
    }

    else if (folder.compare("dpv") == 0) {
      std::tuple<int, int> shape;
      trx->data_per_vertex[base] = std::make_unique<ArraySequence<DT>>();
      int nb_scalar = size / static_cast<int>(trx->header["NB_VERTICES"].int_value());

      if (size % static_cast<int>(trx->header["NB_VERTICES"].int_value()) != 0 || nb_scalar != dim) {

        throw std::invalid_argument("Wrong dpv size/dimensionality");
      } else {
        shape = std::make_tuple(static_cast<int>(trx->header["NB_VERTICES"].int_value()), nb_scalar);
      }
      trx->data_per_vertex[base]->mmap_pos = trx::_create_memmap(filename, shape, "r+", ext, mem_adress);

      if (ext.compare("float16") == 0) {
        new (&(trx->data_per_vertex[base]->_data))
            Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_vertex[base]->mmap_pos.data()),
                                                std::get<0>(shape),
                                                std::get<1>(shape));
      } else if (ext.compare("float32") == 0) {
        new (&(trx->data_per_vertex[base]->_data))
            Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_vertex[base]->mmap_pos.data()),
                                                 std::get<0>(shape),
                                                 std::get<1>(shape));
      } else {
        new (&(trx->data_per_vertex[base]->_data)) Map<Matrix<double, Dynamic, Dynamic>>(
            reinterpret_cast<double *>(trx->data_per_vertex[base]->mmap_pos.data()),
            std::get<0>(shape),
            std::get<1>(shape));
      }

      new (&(trx->data_per_vertex[base]->_offsets))
          Map<Matrix<uint64_t, Dynamic, 1>>(trx->streamlines->_offsets.data(), std::get<0>(shape), std::get<1>(shape));
      trx->data_per_vertex[base]->_lengths = trx->streamlines->_lengths;
    }

    else if (folder.rfind("dpg", 0) == 0) {
      std::tuple<int, int> shape;

      if (size != dim) {

        throw std::invalid_argument("Wrong dpg size/dimensionality");
      } else {
        shape = std::make_tuple(1, static_cast<int>(size));
      }

      std::string data_name = path_basename(base);
      std::string sub_folder = path_basename(folder);

      trx->data_per_group[sub_folder][data_name] = std::make_unique<MMappedMatrix<DT>>();
      trx->data_per_group[sub_folder][data_name]->mmap = trx::_create_memmap(filename, shape, "r+", ext, mem_adress);

      if (ext.compare("float16") == 0) {
        new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(
            reinterpret_cast<half *>(trx->data_per_group[sub_folder][data_name]->mmap.data()),
            std::get<0>(shape),
            std::get<1>(shape));
      } else if (ext.compare("float32") == 0) {
        new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(
            reinterpret_cast<float *>(trx->data_per_group[sub_folder][data_name]->mmap.data()),
            std::get<0>(shape),
            std::get<1>(shape));
      } else {
        new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(
            reinterpret_cast<double *>(trx->data_per_group[sub_folder][data_name]->mmap.data()),
            std::get<0>(shape),
            std::get<1>(shape));
      }
    }

    else if (folder.compare("groups") == 0) {
      std::tuple<int, int> shape;
      if (dim != 1) {
        throw std::invalid_argument("Wrong group dimensionality");
      } else {
        shape = std::make_tuple(static_cast<int>(size), 1);
      }
      trx->groups[base] = std::make_unique<MMappedMatrix<uint32_t>>();
      trx->groups[base]->mmap = trx::_create_memmap(filename, shape, "r+", ext, mem_adress);
      new (&(trx->groups[base]->_matrix)) Map<Matrix<uint32_t, Dynamic, Dynamic>>(
          reinterpret_cast<uint32_t *>(trx->groups[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
    } else {
      throw std::invalid_argument("Entry is not part of a valid TRX structure: " + elem_filename);
    }
  }
  if (trx->streamlines->_data.size() == 0 || trx->streamlines->_offsets.size() == 0) {

    throw std::invalid_argument("Missing essential data.");
  }

  return trx;
}

// TODO: Major refactoring
template <typename DT> std::unique_ptr<TrxFile<DT>> TrxFile<DT>::deepcopy() {
  if (!this->streamlines || this->streamlines->_data.size() == 0 || this->streamlines->_offsets.size() == 0) {
    auto empty_copy = std::make_unique<trx::TrxFile<DT>>();
    empty_copy->header = this->header;
    return empty_copy;
  }
  std::string tmp_dir = make_temp_dir("trx");

  std::string header = tmp_dir + SEPARATOR + "header.json";
  std::ofstream out_json(header);

  // TODO: Definitely a better way to deepcopy
  json tmp_header = this->header;

  auto to_dump = std::make_unique<ArraySequence<DT>>();
  // TODO: Verify that this is indeed a deep copy
  new (&(to_dump->_data)) Matrix<DT, Dynamic, Dynamic, RowMajor>(this->streamlines->_data);
  new (&(to_dump->_offsets)) Matrix<uint64_t, Dynamic, Dynamic, RowMajor>(this->streamlines->_offsets);
  new (&(to_dump->_lengths)) Matrix<uint32_t, Dynamic, 1>(this->streamlines->_lengths);

  if (!this->_copy_safe) {
    const int nb_streamlines = to_dump->_offsets.size() > 0 ? static_cast<int>(to_dump->_offsets.size() - 1) : 0;
    const int nb_vertices = static_cast<int>(to_dump->_data.size() / 3);
    tmp_header = _json_set(tmp_header, "NB_STREAMLINES", nb_streamlines);
    tmp_header = _json_set(tmp_header, "NB_VERTICES", nb_vertices);
  }
  // Ensure sentinel is correct before persisting
  if (to_dump->_offsets.size() > 0) {
    to_dump->_offsets(to_dump->_offsets.size() - 1) = static_cast<uint64_t>(tmp_header["NB_VERTICES"].int_value());
  }
  if (out_json.is_open()) {
    out_json << tmp_header.dump() << std::endl;
    out_json.close();
  }

  std::string pos_rootfn = tmp_dir + SEPARATOR + "positions";
  std::string positions_filename = _generate_filename_from_data(to_dump->_data, pos_rootfn);

  write_binary(positions_filename, to_dump->_data);

  std::string off_rootfn = tmp_dir + SEPARATOR + "offsets";
  std::string offsets_filename = _generate_filename_from_data(to_dump->_offsets, off_rootfn);

  write_binary(offsets_filename, to_dump->_offsets);

  if (this->data_per_vertex.size() > 0) {
    std::string dpv_dirname = tmp_dir + SEPARATOR + "dpv" + SEPARATOR;
    {
      std::error_code ec;
      trx::fs::create_directories(dpv_dirname, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + dpv_dirname);
      }
    }
    for (auto const &x : this->data_per_vertex) {
      Matrix<DT, Dynamic, Dynamic, RowMajor> dpv_todump = x.second->_data;
      std::string dpv_filename = dpv_dirname + x.first;
      dpv_filename = _generate_filename_from_data(dpv_todump, dpv_filename);

      write_binary(dpv_filename, dpv_todump);
    }
  }

  if (this->data_per_streamline.size() > 0) {
    std::string dps_dirname = tmp_dir + SEPARATOR + "dps" + SEPARATOR;
    {
      std::error_code ec;
      trx::fs::create_directories(dps_dirname, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + dps_dirname);
      }
    }
    for (auto const &x : this->data_per_streamline) {
      Matrix<DT, Dynamic, Dynamic> dps_todump = x.second->_matrix;
      std::string dps_filename = dps_dirname + x.first;
      dps_filename = _generate_filename_from_data(dps_todump, dps_filename);

      write_binary(dps_filename, dps_todump);
    }
  }

  if (this->groups.size() > 0) {
    std::string groups_dirname = tmp_dir + SEPARATOR + "groups" + SEPARATOR;
    {
      std::error_code ec;
      trx::fs::create_directories(groups_dirname, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + groups_dirname);
      }
    }

    for (auto const &x : this->groups) {
      Matrix<uint32_t, Dynamic, Dynamic> group_todump = x.second->_matrix;
      std::string group_filename = groups_dirname + x.first;
      group_filename = _generate_filename_from_data(group_todump, group_filename);

      write_binary(group_filename, group_todump);

      if (this->data_per_group.find(x.first) == this->data_per_group.end()) {
        continue;
      }

      for (auto const &y : this->data_per_group[x.first]) {
        std::string dpg_dirname = tmp_dir + SEPARATOR + "dpg" + SEPARATOR;
        std::string dpg_subdirname = dpg_dirname + x.first;
        std::error_code ec;
        if (!trx::fs::exists(dpg_dirname, ec)) {
          ec.clear();
          trx::fs::create_directories(dpg_dirname, ec);
        }
        if (ec) {
          throw std::runtime_error("Could not create directory " + dpg_dirname);
        }
        ec.clear();
        if (!trx::fs::exists(dpg_subdirname, ec)) {
          ec.clear();
          trx::fs::create_directories(dpg_subdirname, ec);
        }
        if (ec) {
          throw std::runtime_error("Could not create directory " + dpg_subdirname);
        }

        Matrix<DT, Dynamic, Dynamic> dpg_todump = this->data_per_group[x.first][y.first]->_matrix;
        std::string dpg_filename = dpg_subdirname + SEPARATOR + y.first;
        dpg_filename = _generate_filename_from_data(dpg_todump, dpg_filename);

        write_binary(dpg_filename, dpg_todump);
      }
    }
  }

  auto copy_trx = TrxFile<DT>::load_from_directory(tmp_dir);
  copy_trx->_uncompressed_folder_handle = tmp_dir;
  copy_trx->_owns_uncompressed_folder = true;

  return copy_trx;
}

// TODO: verify that this function is actually necessary (there should not be preallocation zeros
// afaik)
template <typename DT> std::tuple<int, int> TrxFile<DT>::_get_real_len() {
  if (this->streamlines->_lengths.size() == 0)
    return std::make_tuple(0, 0);

  int last_elem_pos = trx::detail::_dichotomic_search(this->streamlines->_lengths);

  if (last_elem_pos != -1) {
    int strs_end = last_elem_pos + 1;
    int pts_end = this->streamlines->_lengths(Eigen::seq(0, last_elem_pos), 0).sum();

    return std::make_tuple(strs_end, pts_end);
  }

  return std::make_tuple(0, 0);
}

template <typename DT>
std::tuple<int, int>
TrxFile<DT>::_copy_fixed_arrays_from(TrxFile<DT> *trx, int strs_start, int pts_start, int nb_strs_to_copy) {
  int curr_strs_len, curr_pts_len;

  if (nb_strs_to_copy == -1) {
    std::tuple<int, int> curr = this->_get_real_len();
    curr_strs_len = std::get<0>(curr);
    curr_pts_len = std::get<1>(curr);
  } else {
    curr_strs_len = nb_strs_to_copy;
    curr_pts_len = trx->streamlines->_lengths(Eigen::seq(0, curr_strs_len - 1)).sum();
  }

  if (pts_start == -1) {
    pts_start = 0;
  }
  if (strs_start == -1) {
    strs_start = 0;
  }

  int strs_end = strs_start + curr_strs_len;
  int pts_end = pts_start + curr_pts_len;

  if (curr_pts_len == 0)
    return std::make_tuple(strs_start, pts_start);

  this->streamlines->_data.block(pts_start, 0, curr_pts_len, this->streamlines->_data.cols()) =
      trx->streamlines->_data.block(0, 0, curr_pts_len, trx->streamlines->_data.cols());

  this->streamlines->_offsets.block(strs_start, 0, curr_strs_len + 1, 1) =
      (trx->streamlines->_offsets.block(0, 0, curr_strs_len + 1, 1).array() + pts_start).matrix();

  this->streamlines->_lengths.block(strs_start, 0, curr_strs_len, 1) =
      trx->streamlines->_lengths.block(0, 0, curr_strs_len, 1);

  for (auto const &x : this->data_per_vertex) {
    this->data_per_vertex[x.first]->_data.block(
        pts_start, 0, curr_pts_len, this->data_per_vertex[x.first]->_data.cols()) =
        trx->data_per_vertex[x.first]->_data.block(0, 0, curr_pts_len, trx->data_per_vertex[x.first]->_data.cols());
    new (&(this->data_per_vertex[x.first]->_offsets))
        Map<Matrix<uint64_t, Dynamic, Dynamic>>(trx->data_per_vertex[x.first]->_offsets.data(),
                                                trx->data_per_vertex[x.first]->_offsets.rows(),
                                                trx->data_per_vertex[x.first]->_offsets.cols());
    this->data_per_vertex[x.first]->_lengths = trx->data_per_vertex[x.first]->_lengths;
  }

  for (auto const &x : this->data_per_streamline) {
    this->data_per_streamline[x.first]->_matrix.block(
        strs_start, 0, curr_strs_len, this->data_per_streamline[x.first]->_matrix.cols()) =
        trx->data_per_streamline[x.first]->_matrix.block(
            0, 0, curr_strs_len, trx->data_per_streamline[x.first]->_matrix.cols());
  }

  return std::make_tuple(strs_end, pts_end);
}

template <typename DT> void TrxFile<DT>::close() {
  this->_cleanup_temporary_directory();
  this->streamlines.reset();
  this->groups.clear();
  this->data_per_streamline.clear();
  this->data_per_vertex.clear();
  this->data_per_group.clear();
  this->_uncompressed_folder_handle.clear();
  this->_owns_uncompressed_folder = false;
  this->_copy_safe = true;

  std::vector<std::vector<float>> affine(4, std::vector<float>(4, 0.0f));
  for (int i = 0; i < 4; i++) {
    affine[i][i] = 1.0f;
  }
  std::vector<uint16_t> dimensions{1, 1, 1};
  json::object header_obj;
  header_obj["VOXEL_TO_RASMM"] = affine;
  header_obj["DIMENSIONS"] = dimensions;
  header_obj["NB_VERTICES"] = 0;
  header_obj["NB_STREAMLINES"] = 0;
  this->header = json(header_obj);
}

template <typename DT> TrxFile<DT>::~TrxFile() { this->_cleanup_temporary_directory(); }

template <typename DT>
// Caveat: cleanup is best-effort; filesystem errors are ignored.
void TrxFile<DT>::_cleanup_temporary_directory() {
  if (this->_owns_uncompressed_folder && !this->_uncompressed_folder_handle.empty()) {
    if (rm_dir(this->_uncompressed_folder_handle) != 0) {
    }
    this->_uncompressed_folder_handle.clear();
    this->_owns_uncompressed_folder = false;
  }
}

template <typename DT>
// Caveats: downsizing vertices is not supported; reducing streamlines truncates data; same-size
// resize is a no-op.
void TrxFile<DT>::resize(int nb_streamlines, int nb_vertices, bool delete_dpg) {
  if (!this->_copy_safe) {
    throw std::invalid_argument("Cannot resize a sliced dataset.");
  }

  std::tuple<int, int> sp_end = this->_get_real_len();
  int strs_end = std::get<0>(sp_end);
  int ptrs_end = std::get<1>(sp_end);

  if (nb_streamlines != -1 && nb_streamlines < strs_end) {
    strs_end = nb_streamlines;
  }

  if (nb_vertices == -1) {
    ptrs_end = this->streamlines->_lengths.sum();
    nb_vertices = ptrs_end;
  } else if (nb_vertices < ptrs_end) {
    return;
  }

  if (nb_streamlines == -1) {
    nb_streamlines = strs_end;
  }

  if (nb_streamlines == this->header["NB_STREAMLINES"].int_value() &&
      nb_vertices == this->header["NB_VERTICES"].int_value()) {
    return;
  }

  auto trx = _initialize_empty_trx(nb_streamlines, nb_vertices, this);

  if (nb_streamlines < this->header["NB_STREAMLINES"].int_value())
    trx->_copy_fixed_arrays_from(this, -1, -1, nb_streamlines);
  else {
    trx->_copy_fixed_arrays_from(this);
  }

  std::string tmp_dir = trx->_uncompressed_folder_handle;

  if (this->groups.size() > 0) {
    std::string group_dir = tmp_dir + SEPARATOR + "groups" + SEPARATOR;
    {
      std::error_code ec;
      trx::fs::create_directories(group_dir, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + group_dir);
      }
    }

    for (auto const &x : this->groups) {
      std::string group_dtype = dtype_from_scalar<uint32_t>();
      std::string group_name = group_dir + x.first + "." + group_dtype;

      int ori_length = this->groups[x.first]->_matrix.size();

      std::vector<int> keep_rows;
      std::vector<int> keep_cols = {0};

      // Slicing
      for (int i = 0; i < x.second->_matrix.rows(); ++i) {
        for (int j = 0; j < x.second->_matrix.cols(); ++j) {
          if (static_cast<int>(x.second->_matrix(i, j)) < strs_end) {
            keep_rows.push_back(i);
          }
        }
      }
      // std::cout << "Cols " << keep_rows.at(1) << std::endl;

      Matrix<uint32_t, Dynamic, Dynamic> tmp = this->groups[x.first]->_matrix(keep_rows, keep_cols);
      std::tuple<int, int> group_shape = std::make_tuple(tmp.size(), 1);

      trx->groups[x.first] = std::make_unique<MMappedMatrix<uint32_t>>();
      trx->groups[x.first]->mmap = trx::_create_memmap(group_name, group_shape, "w+", group_dtype);
      new (&(trx->groups[x.first]->_matrix))
          Map<Matrix<uint32_t, Dynamic, Dynamic>>(reinterpret_cast<uint32_t *>(trx->groups[x.first]->mmap.data()),
                                                  std::get<0>(group_shape),
                                                  std::get<1>(group_shape));

      // update values
      for (int i = 0; i < trx->groups[x.first]->_matrix.rows(); ++i) {
        for (int j = 0; j < trx->groups[x.first]->_matrix.cols(); ++j) {
          trx->groups[x.first]->_matrix(i, j) = tmp(i, j);
        }
      }
    }
  }

  if (delete_dpg) {
    this->close();
    return;
  }

  if (this->data_per_group.size() > 0) {
    // really need to refactor all these mkdirs
    std::string dpg_dir = tmp_dir + SEPARATOR + "dpg" + SEPARATOR;
    {
      std::error_code ec;
      trx::fs::create_directories(dpg_dir, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + dpg_dir);
      }
    }

    for (auto const &x : this->data_per_group) {
      std::string dpg_subdir = dpg_dir + x.first;
      {
        std::error_code ec;
        trx::fs::create_directories(dpg_subdir, ec);
        if (ec) {
          throw std::runtime_error("Could not create directory " + dpg_subdir);
        }
      }

      if (trx->data_per_group.find(x.first) == trx->data_per_group.end()) {
        trx->data_per_group.emplace(x.first, std::map<std::string, std::unique_ptr<MMappedMatrix<DT>>>{});
      } else {
        trx->data_per_group[x.first].clear();
      }

      for (auto const &y : this->data_per_group[x.first]) {
        std::string dpg_dtype = dtype_from_scalar<DT>();
        std::string dpg_filename = dpg_subdir + SEPARATOR + y.first;
        dpg_filename = _generate_filename_from_data(this->data_per_group[x.first][y.first]->_matrix, dpg_filename);

        std::tuple<int, int> dpg_shape = std::make_tuple(this->data_per_group[x.first][y.first]->_matrix.rows(),
                                                         this->data_per_group[x.first][y.first]->_matrix.cols());

        if (trx->data_per_group[x.first].find(y.first) == trx->data_per_group[x.first].end()) {
          trx->data_per_group[x.first][y.first] = std::make_unique<MMappedMatrix<DT>>();
        }

        trx->data_per_group[x.first][y.first]->mmap = _create_memmap(dpg_filename, dpg_shape, "w+", dpg_dtype);
        new (&(trx->data_per_group[x.first][y.first]->_matrix)) Map<Matrix<uint32_t, Dynamic, Dynamic>>(
            reinterpret_cast<uint32_t *>(trx->data_per_group[x.first][y.first]->mmap.data()),
            std::get<0>(dpg_shape),
            std::get<1>(dpg_shape));

        // update values
        for (int i = 0; i < trx->data_per_group[x.first][y.first]->_matrix.rows(); ++i) {
          for (int j = 0; j < trx->data_per_group[x.first][y.first]->_matrix.cols(); ++j) {
            trx->data_per_group[x.first][y.first]->_matrix(i, j) =
                this->data_per_group[x.first][y.first]->_matrix(i, j);
          }
        }
      }
    }
    this->close();
  }
}

template <typename DT> std::unique_ptr<TrxFile<DT>> TrxFile<DT>::load_from_zip(const std::string &filename) {
  int errorp = 0;
  zip_t *zf = open_zip_for_read(filename, errorp);
  if (zf == nullptr) {
    throw std::runtime_error("Could not open zip file: " + filename);
  }

  std::string temp_dir = extract_zip_to_directory(zf);
  zip_close(zf);

  auto trx = TrxFile<DT>::load_from_directory(temp_dir);
  trx->_uncompressed_folder_handle = temp_dir;
  trx->_owns_uncompressed_folder = true;
  return trx;
}

template <typename DT> std::unique_ptr<TrxFile<DT>> TrxFile<DT>::load_from_directory(const std::string &path) {
  std::string directory = path;
  {
    std::error_code ec;
    trx::fs::path resolved = trx::fs::weakly_canonical(trx::fs::path(path), ec);
    if (!ec) {
      directory = resolved.string();
    }
  }
  std::string header_name = directory + SEPARATOR + "header.json";

  // TODO: add check to verify that it's open
  std::ifstream header_file;
  for (int attempt = 0; attempt < 5; ++attempt) {
    header_file.open(header_name);
    if (header_file.is_open()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!header_file.is_open()) {
    std::error_code ec;
    const bool exists = trx::fs::exists(directory, ec);
    const int open_err = errno;
    std::string detail = "Failed to open header.json at: " + header_name;
    detail += " exists=" + std::string(exists ? "true" : "false");
    detail += " errno=" + std::to_string(open_err) + " msg=" + std::string(std::strerror(open_err));
    if (exists) {
      std::vector<std::string> files;
      for (const auto &entry : trx::fs::directory_iterator(directory, ec)) {
        if (ec) {
          break;
        }
        files.push_back(entry.path().filename().string());
      }
      if (!files.empty()) {
        std::sort(files.begin(), files.end());
        detail += " files=[";
        for (size_t i = 0; i < files.size(); ++i) {
          if (i > 0) {
            detail += ",";
          }
          detail += files[i];
        }
        detail += "]";
      }
    }
    throw std::runtime_error(detail);
  }
  std::string jstream((std::istreambuf_iterator<char>(header_file)), std::istreambuf_iterator<char>());
  header_file.close();
  std::string err;
  json header = json::parse(jstream, err);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse header.json: " + err);
  }

  std::map<std::string, std::tuple<long long, long long>> files_pointer_size;
  populate_fps(directory, files_pointer_size);

  auto trx = TrxFile<DT>::_create_trx_from_pointer(header, files_pointer_size, "", directory);
  trx->_uncompressed_folder_handle = directory;
  trx->_owns_uncompressed_folder = false;
  return trx;
}

template <typename DT> std::unique_ptr<TrxFile<DT>> TrxFile<DT>::load(const std::string &path) {
  trx::fs::path input(path);
  if (!trx::fs::exists(input)) {
    throw std::runtime_error("Input path does not exist: " + path);
  }
  std::error_code ec;
  if (trx::fs::is_directory(input, ec) && !ec) {
    return TrxFile<DT>::load_from_directory(path);
  }
  return TrxFile<DT>::load_from_zip(path);
}

template <typename DT> std::unique_ptr<TrxFile<DT>> load(const std::string &path) {
  return TrxFile<DT>::load(path);
}

template <typename DT> TrxReader<DT>::TrxReader(const std::string &path) { trx_ = TrxFile<DT>::load(path); }

template <typename DT> TrxReader<DT>::TrxReader(TrxReader &&other) noexcept : trx_(std::move(other.trx_)) {}

template <typename DT> TrxReader<DT> &TrxReader<DT>::operator=(TrxReader &&other) noexcept {
  if (this != &other) {
    trx_ = std::move(other.trx_);
  }
  return *this;
}

template <typename Fn>
auto with_trx_reader(const std::string &path, Fn &&fn)
    -> decltype(fn(std::declval<TrxReader<float> &>(), TrxScalarType::Float32)) {
  const TrxScalarType dtype = detect_positions_scalar_type(path, TrxScalarType::Float32);
  switch (dtype) {
  case TrxScalarType::Float16: {
    TrxReader<Eigen::half> reader(path);
    return fn(reader, dtype);
  }
  case TrxScalarType::Float64: {
    TrxReader<double> reader(path);
    return fn(reader, dtype);
  }
  case TrxScalarType::Float32:
  default: {
    TrxReader<float> reader(path);
    return fn(reader, dtype);
  }
  }
}

template <typename DT> void TrxFile<DT>::save(const std::string &filename, zip_uint32_t compression_standard) {
  TrxSaveOptions options;
  options.compression_standard = compression_standard;
  save(filename, options);
}

template <typename DT> void TrxFile<DT>::normalize_for_save() {
  if (!this->streamlines) {
    throw std::runtime_error("Cannot normalize TRX without streamline data");
  }
  if (this->streamlines->_offsets.size() == 0) {
    throw std::runtime_error("Cannot normalize TRX without offsets data");
  }

  const size_t offsets_count = static_cast<size_t>(this->streamlines->_offsets.size());
  if (offsets_count < 1) {
    throw std::runtime_error("Invalid offsets array");
  }
  const size_t total_streamlines = offsets_count - 1;
  const uint64_t data_rows = static_cast<uint64_t>(this->streamlines->_data.rows());

  size_t used_streamlines = total_streamlines;
  for (size_t i = 1; i < offsets_count; ++i) {
    const uint64_t prev = static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(i - 1)));
    const uint64_t curr = static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(i)));
    if (curr < prev || curr > data_rows) {
      used_streamlines = i - 1;
      break;
    }
  }

  const uint64_t used_vertices =
      static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(used_streamlines)));
  if (used_vertices > data_rows) {
    throw std::runtime_error("TRX offsets exceed positions row count");
  }
  if (used_vertices > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      used_streamlines > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("TRX normalize_for_save exceeds supported int range");
  }

  if (used_streamlines < total_streamlines || used_vertices < data_rows) {
    this->resize(static_cast<int>(used_streamlines), static_cast<int>(used_vertices));
  }

  const size_t normalized_streamlines = static_cast<size_t>(this->streamlines->_offsets.size()) - 1;
  for (size_t i = 0; i < normalized_streamlines; ++i) {
    const uint64_t curr = static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(i)));
    const uint64_t next = static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(i + 1)));
    if (next < curr) {
      throw std::runtime_error("TRX offsets must be monotonically increasing");
    }
    const uint64_t diff = next - curr;
    if (diff > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("TRX streamline length exceeds uint32 range");
    }
    this->streamlines->_lengths(static_cast<Eigen::Index>(i)) = static_cast<uint32_t>(diff);
  }

  const uint64_t sentinel = static_cast<uint64_t>(
      this->streamlines->_offsets(static_cast<Eigen::Index>(this->streamlines->_offsets.size() - 1)));
  this->header = _json_set(this->header, "NB_STREAMLINES", static_cast<int>(normalized_streamlines));
  this->header = _json_set(this->header, "NB_VERTICES", static_cast<int>(sentinel));
}

template <typename DT> void TrxFile<DT>::save(const std::string &filename, const TrxSaveOptions &options) {
  std::string ext = get_ext(filename);

  if (ext.size() > 0 && ext != "zip" && ext != "trx") {
    throw std::invalid_argument("Unsupported extension: " + ext);
  }

  TrxFile<DT> *save_trx = this;

  if (!save_trx->streamlines || save_trx->streamlines->_offsets.size() == 0) {
    throw std::runtime_error("Cannot save TRX without offsets data");
  }
  if (save_trx->header["NB_STREAMLINES"].is_number()) {
    const auto nb_streamlines = static_cast<size_t>(save_trx->header["NB_STREAMLINES"].int_value());
    if (save_trx->streamlines->_offsets.size() != static_cast<Eigen::Index>(nb_streamlines + 1)) {
      throw std::runtime_error("TRX offsets size does not match NB_STREAMLINES");
    }
  }
  if (save_trx->header["NB_VERTICES"].is_number()) {
    const auto nb_vertices = static_cast<uint64_t>(save_trx->header["NB_VERTICES"].int_value());
    const auto last =
        static_cast<uint64_t>(save_trx->streamlines->_offsets(save_trx->streamlines->_offsets.size() - 1));
    if (last != nb_vertices) {
      throw std::runtime_error("TRX offsets sentinel does not match NB_VERTICES");
    }
  }
  for (Eigen::Index i = 1; i < save_trx->streamlines->_offsets.size(); ++i) {
    if (save_trx->streamlines->_offsets(i) < save_trx->streamlines->_offsets(i - 1)) {
      throw std::runtime_error("TRX offsets must be monotonically increasing");
    }
  }
  if (save_trx->streamlines->_data.size() > 0) {
    const auto last =
        static_cast<uint64_t>(save_trx->streamlines->_offsets(save_trx->streamlines->_offsets.size() - 1));
    if (last != static_cast<uint64_t>(save_trx->streamlines->_data.rows())) {
      throw std::runtime_error("TRX positions row count does not match offsets sentinel");
    }
  }
  std::string tmp_dir_name = save_trx->_uncompressed_folder_handle;

  if (!tmp_dir_name.empty()) {
    const std::string header_path = tmp_dir_name + SEPARATOR + "header.json";
    std::ofstream out_json(header_path, std::ios::out | std::ios::trunc);
    if (!out_json.is_open()) {
      throw std::runtime_error("Failed to write header.json to: " + header_path);
    }
    out_json << save_trx->header.dump() << std::endl;
    out_json.close();
  }

  const bool write_archive = options.mode == TrxSaveMode::Archive ||
                             (options.mode == TrxSaveMode::Auto && ext.size() > 0 && (ext == "zip" || ext == "trx"));
  if (write_archive) {
    auto sync_unmap_seq = [&](auto &seq) {
      if (!seq) {
        return;
      }
      std::error_code ec;
      seq->mmap_pos.sync(ec);
      seq->mmap_off.sync(ec);
    };
    auto sync_unmap_mat = [&](auto &mat) {
      if (!mat) {
        return;
      }
      std::error_code ec;
      mat->mmap.sync(ec);
    };

    sync_unmap_seq(save_trx->streamlines);
    for (auto &kv : save_trx->groups) {
      sync_unmap_mat(kv.second);
    }
    for (auto &kv : save_trx->data_per_streamline) {
      sync_unmap_mat(kv.second);
    }
    for (auto &kv : save_trx->data_per_vertex) {
      sync_unmap_seq(kv.second);
    }
    for (auto &group_kv : save_trx->data_per_group) {
      for (auto &kv : group_kv.second) {
        sync_unmap_mat(kv.second);
      }
    }

    int errorp;
    zip_t *zf;
    if ((zf = zip_open(filename.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp)) == nullptr) {
      throw std::runtime_error("Could not open archive " + filename + ": " + strerror(errorp));
    } else {
      zip_from_folder(zf, tmp_dir_name, tmp_dir_name, options.compression_standard, nullptr);
      if (zip_close(zf) != 0) {
        throw std::runtime_error("Unable to close archive " + filename + ": " + zip_strerror(zf));
      }
    }
  } else {
    std::error_code ec;
    if (!trx::fs::exists(tmp_dir_name, ec) || !trx::fs::is_directory(tmp_dir_name, ec)) {
      throw std::runtime_error("Temporary TRX directory does not exist: " + tmp_dir_name);
    }
    if (trx::fs::exists(filename, ec) && trx::fs::is_directory(filename, ec)) {
      if (!options.overwrite_existing) {
        throw std::runtime_error("Output directory already exists: " + filename);
      }
      if (rm_dir(filename) != 0) {
        throw std::runtime_error("Could not remove existing directory " + filename);
      }
    }
    trx::fs::path dest_path(filename);
    if (dest_path.has_parent_path()) {
      std::error_code ec;
      trx::fs::create_directories(dest_path.parent_path(), ec);
      if (ec) {
        throw std::runtime_error("Could not create output parent directory: " + dest_path.parent_path().string());
      }
    }
    copy_dir(tmp_dir_name, filename);
    ec.clear();
    if (!trx::fs::exists(filename, ec) || !trx::fs::is_directory(filename, ec)) {
      throw std::runtime_error("Failed to create output directory: " + filename);
    }
    const trx::fs::path header_path = dest_path / "header.json";
    if (!trx::fs::exists(header_path)) {
      throw std::runtime_error("Missing header.json in output directory: " + header_path.string());
    }
  }
}

template <typename DT>
void TrxFile<DT>::add_dps_from_text(const std::string &name, const std::string &dtype, const std::string &path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open DPS text file: " + path);
  }

  std::vector<double> values;
  double value = 0.0;
  while (input >> value) {
    values.push_back(value);
  }
  if (!input.eof() && input.fail()) {
    throw std::runtime_error("Failed to parse DPS text file: " + path);
  }

  add_dps_from_vector(name, dtype, values);
}

template <typename DT>
template <typename T>
void TrxFile<DT>::add_dps_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values) {
  if (name.empty()) {
    throw std::invalid_argument("DPS name cannot be empty");
  }

  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (!trx::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported DPS dtype: " + dtype);
  }
  if (dtype_norm != "float16" && dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported DPS dtype: " + dtype);
  }

  if (this->_uncompressed_folder_handle.empty()) {
    throw std::runtime_error("TRX file has no backing directory to store DPS data");
  }

  size_t nb_streamlines = 0;
  if (this->streamlines) {
    nb_streamlines = static_cast<size_t>(this->streamlines->_lengths.size());
  } else if (this->header["NB_STREAMLINES"].is_number()) {
    nb_streamlines = static_cast<size_t>(this->header["NB_STREAMLINES"].int_value());
  }

  if (values.size() != nb_streamlines) {
    throw std::runtime_error("DPS values (" + std::to_string(values.size()) + ") do not match number of streamlines (" +
                             std::to_string(nb_streamlines) + ")");
  }

  std::string dps_dirname = this->_uncompressed_folder_handle + SEPARATOR + "dps" + SEPARATOR;
  {
    std::error_code ec;
    trx::fs::create_directories(dps_dirname, ec);
    if (ec) {
      throw std::runtime_error("Could not create directory " + dps_dirname);
    }
  }

  std::string dps_filename = dps_dirname + name + "." + dtype_norm;
  {
    std::error_code ec;
    if (trx::fs::exists(dps_filename, ec)) {
      trx::fs::remove(dps_filename, ec);
    }
  }

  auto existing = this->data_per_streamline.find(name);
  if (existing != this->data_per_streamline.end()) {
    this->data_per_streamline.erase(existing);
  }

  const int rows = static_cast<int>(nb_streamlines);
  const int cols = 1;
  std::tuple<int, int> shape = std::make_tuple(rows, cols);

  auto matrix = std::make_unique<trx::MMappedMatrix<DT>>();
  matrix->mmap = trx::_create_memmap(dps_filename, shape, "w+", dtype_norm);

  if (dtype_norm == "float16") {
    auto *data = reinterpret_cast<half *>(matrix->mmap.data());
    Map<Matrix<half, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(matrix->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<half>(values[static_cast<size_t>(i)]);
    }
  } else if (dtype_norm == "float32") {
    auto *data = reinterpret_cast<float *>(matrix->mmap.data());
    Map<Matrix<float, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(matrix->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<float>(values[static_cast<size_t>(i)]);
    }
  } else {
    auto *data = reinterpret_cast<double *>(matrix->mmap.data());
    Map<Matrix<double, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(matrix->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<double>(values[static_cast<size_t>(i)]);
    }
  }

  this->data_per_streamline[name] = std::move(matrix);
}

template <typename DT>
template <typename T>
void TrxFile<DT>::add_dpv_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values) {
  if (name.empty()) {
    throw std::invalid_argument("DPV name cannot be empty");
  }

  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (!trx::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported DPV dtype: " + dtype);
  }
  if (dtype_norm != "float16" && dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported DPV dtype: " + dtype);
  }

  if (this->_uncompressed_folder_handle.empty()) {
    throw std::runtime_error("TRX file has no backing directory to store DPV data");
  }

  size_t nb_vertices = 0;
  if (this->streamlines) {
    nb_vertices = static_cast<size_t>(this->streamlines->_data.rows());
  } else if (this->header["NB_VERTICES"].is_number()) {
    nb_vertices = static_cast<size_t>(this->header["NB_VERTICES"].int_value());
  }

  if (values.size() != nb_vertices) {
    throw std::runtime_error("DPV values (" + std::to_string(values.size()) + ") do not match number of vertices (" +
                             std::to_string(nb_vertices) + ")");
  }

  std::string dpv_dirname = this->_uncompressed_folder_handle + SEPARATOR + "dpv" + SEPARATOR;
  {
    std::error_code ec;
    trx::fs::create_directories(dpv_dirname, ec);
    if (ec) {
      throw std::runtime_error("Could not create directory " + dpv_dirname);
    }
  }

  std::string dpv_filename = dpv_dirname + name + "." + dtype_norm;
  {
    std::error_code ec;
    if (trx::fs::exists(dpv_filename, ec)) {
      trx::fs::remove(dpv_filename, ec);
    }
  }

  auto existing = this->data_per_vertex.find(name);
  if (existing != this->data_per_vertex.end()) {
    this->data_per_vertex.erase(existing);
  }

  const int rows = static_cast<int>(nb_vertices);
  const int cols = 1;
  std::tuple<int, int> shape = std::make_tuple(rows, cols);

  auto seq = std::make_unique<trx::ArraySequence<DT>>();
  seq->mmap_pos = trx::_create_memmap(dpv_filename, shape, "w+", dtype_norm);

  if (dtype_norm == "float16") {
    auto *data = reinterpret_cast<half *>(seq->mmap_pos.data());
    Map<Matrix<half, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(seq->_data)) Map<Matrix<half, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<half>(values[static_cast<size_t>(i)]);
    }
  } else if (dtype_norm == "float32") {
    auto *data = reinterpret_cast<float *>(seq->mmap_pos.data());
    Map<Matrix<float, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(seq->_data)) Map<Matrix<float, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<float>(values[static_cast<size_t>(i)]);
    }
  } else {
    auto *data = reinterpret_cast<double *>(seq->mmap_pos.data());
    Map<Matrix<double, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(seq->_data)) Map<Matrix<double, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<double>(values[static_cast<size_t>(i)]);
    }
  }

  if (this->streamlines && this->streamlines->_offsets.size() > 0) {
    new (&(seq->_offsets)) Map<Matrix<uint64_t, Dynamic, Dynamic>>(this->streamlines->_offsets.data(),
                                                                   int(this->streamlines->_offsets.rows()),
                                                                   int(this->streamlines->_offsets.cols()));
    seq->_lengths = this->streamlines->_lengths;
  }

  this->data_per_vertex[name] = std::move(seq);
}

template <typename DT>
void TrxFile<DT>::add_group_from_indices(const std::string &name, const std::vector<uint32_t> &indices) {
  if (name.empty()) {
    throw std::invalid_argument("Group name cannot be empty");
  }
  if (this->_uncompressed_folder_handle.empty()) {
    throw std::runtime_error("TRX file has no backing directory to store groups");
  }

  size_t nb_streamlines = 0;
  if (this->streamlines) {
    nb_streamlines = static_cast<size_t>(this->streamlines->_lengths.size());
  } else if (this->header["NB_STREAMLINES"].is_number()) {
    nb_streamlines = static_cast<size_t>(this->header["NB_STREAMLINES"].int_value());
  }

  for (const auto idx : indices) {
    if (idx >= nb_streamlines) {
      throw std::runtime_error("Group index out of range: " + std::to_string(idx));
    }
  }

  std::string groups_dirname = this->_uncompressed_folder_handle + SEPARATOR + "groups" + SEPARATOR;
  {
    std::error_code ec;
    trx::fs::create_directories(groups_dirname, ec);
    if (ec) {
      throw std::runtime_error("Could not create directory " + groups_dirname);
    }
  }

  std::string group_filename = groups_dirname + name + ".uint32";
  {
    std::error_code ec;
    if (trx::fs::exists(group_filename, ec)) {
      trx::fs::remove(group_filename, ec);
    }
  }

  auto existing = this->groups.find(name);
  if (existing != this->groups.end()) {
    this->groups.erase(existing);
  }

  const int rows = static_cast<int>(indices.size());
  const int cols = 1;
  std::tuple<int, int> shape = std::make_tuple(rows, cols);

  auto group = std::make_unique<trx::MMappedMatrix<uint32_t>>();
  group->mmap = trx::_create_memmap(group_filename, shape, "w+", "uint32");
  new (&(group->_matrix)) Map<Matrix<uint32_t, Dynamic, Dynamic>>(
      reinterpret_cast<uint32_t *>(group->mmap.data()), std::get<0>(shape), std::get<1>(shape));
  for (int i = 0; i < rows; ++i) {
    group->_matrix(i, 0) = indices[static_cast<size_t>(i)];
  }
  this->groups[name] = std::move(group);
}

template <typename DT>
void TrxFile<DT>::set_voxel_to_rasmm(const Eigen::Matrix4f &affine) {
  std::vector<std::vector<float>> matrix(4, std::vector<float>(4, 0.0f));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] = affine(i, j);
    }
  }
  this->header = _json_set(this->header, "VOXEL_TO_RASMM", matrix);
}

inline TrxStream::TrxStream(std::string positions_dtype) : positions_dtype_(std::move(positions_dtype)) {
  std::transform(positions_dtype_.begin(), positions_dtype_.end(), positions_dtype_.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (positions_dtype_ != "float32" && positions_dtype_ != "float16") {
    throw std::invalid_argument("TrxStream only supports float16/float32 positions for now");
  }
  tmp_dir_ = make_temp_dir("trx_proto");
  positions_path_ = tmp_dir_ + SEPARATOR + "positions.tmp";
  ensure_positions_stream();
}

inline TrxStream::~TrxStream() { cleanup_tmp(); }

inline void TrxStream::set_metadata_mode(MetadataMode mode) {
  if (finalized_) {
    throw std::runtime_error("Cannot adjust metadata mode after finalize");
  }
  metadata_mode_ = mode;
}

inline void TrxStream::set_metadata_buffer_max_bytes(std::size_t max_bytes) {
  if (finalized_) {
    throw std::runtime_error("Cannot adjust metadata buffer after finalize");
  }
  metadata_buffer_max_bytes_ = max_bytes;
}

inline void TrxStream::ensure_positions_stream() {
  if (!positions_out_.is_open()) {
    positions_out_.open(positions_path_, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!positions_out_.is_open()) {
      throw std::runtime_error("Failed to open TrxStream temp positions file: " + positions_path_);
    }
  }
}

inline void TrxStream::ensure_metadata_dir(const std::string &subdir) {
  if (tmp_dir_.empty()) {
    throw std::runtime_error("TrxStream temp directory not initialized");
  }
  const std::string dir = tmp_dir_ + SEPARATOR + subdir + SEPARATOR;
  std::error_code ec;
  trx::fs::create_directories(dir, ec);
  if (ec) {
    throw std::runtime_error("Could not create directory " + dir);
  }
}

inline void TrxStream::flush_positions_buffer() {
  if (positions_dtype_ == "float16") {
    if (positions_buffer_half_.empty()) {
      return;
    }
    ensure_positions_stream();
    const size_t byte_count = positions_buffer_half_.size() * sizeof(half);
    positions_out_.write(reinterpret_cast<const char *>(positions_buffer_half_.data()),
                         static_cast<std::streamsize>(byte_count));
    if (!positions_out_) {
      throw std::runtime_error("Failed to write TrxStream positions buffer");
    }
    positions_buffer_half_.clear();
    return;
  }

  if (positions_buffer_float_.empty()) {
    return;
  }
  ensure_positions_stream();
  const size_t byte_count = positions_buffer_float_.size() * sizeof(float);
  positions_out_.write(reinterpret_cast<const char *>(positions_buffer_float_.data()),
                       static_cast<std::streamsize>(byte_count));
  if (!positions_out_) {
    throw std::runtime_error("Failed to write TrxStream positions buffer");
  }
  positions_buffer_float_.clear();
}

inline void TrxStream::cleanup_tmp() {
  positions_buffer_float_.clear();
  positions_buffer_half_.clear();
  if (positions_out_.is_open()) {
    positions_out_.close();
  }
  if (!tmp_dir_.empty()) {
    rm_dir(tmp_dir_);
    tmp_dir_.clear();
  }
}

inline void TrxStream::push_streamline(const float *xyz, size_t point_count) {
  if (finalized_) {
    throw std::runtime_error("TrxStream already finalized");
  }
  if (point_count == 0) {
    lengths_.push_back(0);
    return;
  }
  if (positions_buffer_max_entries_ == 0) {
    ensure_positions_stream();
    if (positions_dtype_ == "float16") {
      std::vector<half> tmp;
      tmp.reserve(point_count * 3);
      for (size_t i = 0; i < point_count * 3; ++i) {
        tmp.push_back(static_cast<half>(xyz[i]));
      }
      const size_t byte_count = tmp.size() * sizeof(half);
      positions_out_.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(byte_count));
      if (!positions_out_) {
        throw std::runtime_error("Failed to write TrxStream positions");
      }
    } else {
      const size_t byte_count = point_count * 3 * sizeof(float);
      positions_out_.write(reinterpret_cast<const char *>(xyz), static_cast<std::streamsize>(byte_count));
      if (!positions_out_) {
        throw std::runtime_error("Failed to write TrxStream positions");
      }
    }
  } else {
    const size_t floats_count = point_count * 3;
    if (positions_dtype_ == "float16") {
      positions_buffer_half_.reserve(positions_buffer_half_.size() + floats_count);
      for (size_t i = 0; i < floats_count; ++i) {
        positions_buffer_half_.push_back(static_cast<half>(xyz[i]));
      }
      if (positions_buffer_half_.size() >= positions_buffer_max_entries_) {
        flush_positions_buffer();
      }
    } else {
      positions_buffer_float_.insert(positions_buffer_float_.end(), xyz, xyz + floats_count);
      if (positions_buffer_float_.size() >= positions_buffer_max_entries_) {
        flush_positions_buffer();
      }
    }
  }
  total_vertices_ += point_count;
  lengths_.push_back(static_cast<uint32_t>(point_count));
}

inline void TrxStream::push_streamline(const std::vector<float> &xyz_flat) {
  if (xyz_flat.size() % 3 != 0) {
    throw std::invalid_argument("TrxStream streamline buffer must be a multiple of 3");
  }
  push_streamline(xyz_flat.data(), xyz_flat.size() / 3);
}

inline void TrxStream::push_streamline(const std::vector<std::array<float, 3>> &points) {
  if (points.empty()) {
    push_streamline(static_cast<const float *>(nullptr), 0);
    return;
  }
  std::vector<float> xyz_flat;
  xyz_flat.reserve(points.size() * 3);
  for (const auto &point : points) {
    xyz_flat.push_back(point[0]);
    xyz_flat.push_back(point[1]);
    xyz_flat.push_back(point[2]);
  }
  push_streamline(xyz_flat);
}

inline void TrxStream::set_voxel_to_rasmm(const Eigen::Matrix4f &affine) {
  std::vector<std::vector<float>> matrix(4, std::vector<float>(4, 0.0f));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] = affine(i, j);
    }
  }
  header = _json_set(header, "VOXEL_TO_RASMM", matrix);
}

inline void TrxStream::set_dimensions(const std::array<uint16_t, 3> &dims) {
  header = _json_set(header, "DIMENSIONS", std::vector<uint16_t>{dims[0], dims[1], dims[2]});
}

template <typename T>
inline void
TrxStream::push_dps_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values) {
  if (name.empty()) {
    throw std::invalid_argument("DPS name cannot be empty");
  }
  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (!trx::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported DPS dtype: " + dtype);
  }
  if (dtype_norm != "float16" && dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported DPS dtype: " + dtype);
  }
  if (metadata_mode_ == MetadataMode::OnDisk) {
    ensure_metadata_dir("dps");
    const std::string filename = tmp_dir_ + SEPARATOR + "dps" + SEPARATOR + name + "." + dtype_norm;
    std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open DPS file: " + filename);
    }
    if (dtype_norm == "float16") {
      const size_t chunk_elems = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(half));
      std::vector<half> tmp;
      tmp.reserve(chunk_elems);
      size_t offset = 0;
      while (offset < values.size()) {
        const size_t count = std::min(chunk_elems, values.size() - offset);
        tmp.clear();
        for (size_t i = 0; i < count; ++i) {
          tmp.push_back(static_cast<half>(values[offset + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(count * sizeof(half)));
        offset += count;
      }
    } else if (dtype_norm == "float32") {
      const size_t chunk_elems = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(float));
      std::vector<float> tmp;
      tmp.reserve(chunk_elems);
      size_t offset = 0;
      while (offset < values.size()) {
        const size_t count = std::min(chunk_elems, values.size() - offset);
        tmp.clear();
        for (size_t i = 0; i < count; ++i) {
          tmp.push_back(static_cast<float>(values[offset + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(count * sizeof(float)));
        offset += count;
      }
    } else {
      const size_t chunk_elems = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(double));
      std::vector<double> tmp;
      tmp.reserve(chunk_elems);
      size_t offset = 0;
      while (offset < values.size()) {
        const size_t count = std::min(chunk_elems, values.size() - offset);
        tmp.clear();
        for (size_t i = 0; i < count; ++i) {
          tmp.push_back(static_cast<double>(values[offset + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(count * sizeof(double)));
        offset += count;
      }
    }
    out.close();
    metadata_files_.push_back({std::string("dps") + SEPARATOR + name + "." + dtype_norm, filename});
  } else {
    FieldValues field;
    field.dtype = dtype_norm;
    field.values.reserve(values.size());
    for (const auto &v : values) {
      field.values.push_back(static_cast<double>(v));
    }
    dps_[name] = std::move(field);
  }
}

template <typename T>
inline void
TrxStream::push_dpv_from_vector(const std::string &name, const std::string &dtype, const std::vector<T> &values) {
  if (name.empty()) {
    throw std::invalid_argument("DPV name cannot be empty");
  }
  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (!trx::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported DPV dtype: " + dtype);
  }
  if (dtype_norm != "float16" && dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported DPV dtype: " + dtype);
  }
  if (metadata_mode_ == MetadataMode::OnDisk) {
    ensure_metadata_dir("dpv");
    const std::string filename = tmp_dir_ + SEPARATOR + "dpv" + SEPARATOR + name + "." + dtype_norm;
    std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open DPV file: " + filename);
    }
    if (dtype_norm == "float16") {
      const size_t chunk_elems = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(half));
      std::vector<half> tmp;
      tmp.reserve(chunk_elems);
      size_t offset = 0;
      while (offset < values.size()) {
        const size_t count = std::min(chunk_elems, values.size() - offset);
        tmp.clear();
        for (size_t i = 0; i < count; ++i) {
          tmp.push_back(static_cast<half>(values[offset + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(count * sizeof(half)));
        offset += count;
      }
    } else if (dtype_norm == "float32") {
      const size_t chunk_elems = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(float));
      std::vector<float> tmp;
      tmp.reserve(chunk_elems);
      size_t offset = 0;
      while (offset < values.size()) {
        const size_t count = std::min(chunk_elems, values.size() - offset);
        tmp.clear();
        for (size_t i = 0; i < count; ++i) {
          tmp.push_back(static_cast<float>(values[offset + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(count * sizeof(float)));
        offset += count;
      }
    } else {
      const size_t chunk_elems = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(double));
      std::vector<double> tmp;
      tmp.reserve(chunk_elems);
      size_t offset = 0;
      while (offset < values.size()) {
        const size_t count = std::min(chunk_elems, values.size() - offset);
        tmp.clear();
        for (size_t i = 0; i < count; ++i) {
          tmp.push_back(static_cast<double>(values[offset + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(count * sizeof(double)));
        offset += count;
      }
    }
    out.close();
    metadata_files_.push_back({std::string("dpv") + SEPARATOR + name + "." + dtype_norm, filename});
  } else {
    FieldValues field;
    field.dtype = dtype_norm;
    field.values.reserve(values.size());
    for (const auto &v : values) {
      field.values.push_back(static_cast<double>(v));
    }
    dpv_[name] = std::move(field);
  }
}

inline void TrxStream::set_positions_buffer_max_bytes(std::size_t max_bytes) {
  if (finalized_) {
    throw std::runtime_error("Cannot adjust buffer after finalize");
  }
  if (max_bytes == 0) {
    positions_buffer_max_entries_ = 0;
    positions_buffer_float_.clear();
    positions_buffer_half_.clear();
    return;
  }
  const std::size_t element_size = positions_dtype_ == "float16" ? sizeof(half) : sizeof(float);
  const std::size_t entries = max_bytes / element_size;
  const std::size_t aligned = (entries / 3) * 3;
  positions_buffer_max_entries_ = aligned;
  if (positions_buffer_max_entries_ == 0) {
    positions_buffer_float_.clear();
    positions_buffer_half_.clear();
  }
}

inline void TrxStream::push_group_from_indices(const std::string &name, const std::vector<uint32_t> &indices) {
  if (name.empty()) {
    throw std::invalid_argument("Group name cannot be empty");
  }
  if (metadata_mode_ == MetadataMode::OnDisk) {
    ensure_metadata_dir("groups");
    const std::string filename = tmp_dir_ + SEPARATOR + "groups" + SEPARATOR + name + ".uint32";
    std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open group file: " + filename);
    }
    const size_t chunk_elems = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(uint32_t));
    size_t offset = 0;
    while (offset < indices.size()) {
      const size_t count = std::min(chunk_elems, indices.size() - offset);
      out.write(reinterpret_cast<const char *>(indices.data() + offset),
                static_cast<std::streamsize>(count * sizeof(uint32_t)));
      offset += count;
    }
    out.close();
    metadata_files_.push_back({std::string("groups") + SEPARATOR + name + ".uint32", filename});
  } else {
    groups_[name] = indices;
  }
}

template <typename DT> void TrxStream::finalize(const std::string &filename, zip_uint32_t compression_standard) {
  if (finalized_) {
    throw std::runtime_error("TrxStream already finalized");
  }
  finalized_ = true;

  flush_positions_buffer();
  if (positions_out_.is_open()) {
    positions_out_.flush();
    positions_out_.close();
  }

  const size_t nb_streamlines = lengths_.size();
  const size_t nb_vertices = total_vertices_;

  TrxFile<DT> trx(static_cast<int>(nb_vertices), static_cast<int>(nb_streamlines));

  json header_out = header;
  header_out = _json_set(header_out, "NB_VERTICES", static_cast<int>(nb_vertices));
  header_out = _json_set(header_out, "NB_STREAMLINES", static_cast<int>(nb_streamlines));
  trx.header = header_out;

  auto &positions = trx.streamlines->_data;
  auto &offsets = trx.streamlines->_offsets;
  auto &lengths = trx.streamlines->_lengths;

  offsets(0, 0) = 0;
  for (size_t i = 0; i < nb_streamlines; ++i) {
    lengths(static_cast<Eigen::Index>(i)) = static_cast<uint32_t>(lengths_[i]);
    offsets(static_cast<Eigen::Index>(i + 1), 0) = offsets(static_cast<Eigen::Index>(i), 0) + lengths_[i];
  }

  std::ifstream in(positions_path_, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open TrxStream temp positions file for read: " + positions_path_);
  }
  for (size_t i = 0; i < nb_vertices; ++i) {
    if (positions_dtype_ == "float16") {
      half xyz[3];
      in.read(reinterpret_cast<char *>(xyz), sizeof(xyz));
      if (!in) {
        throw std::runtime_error("Failed to read TrxStream positions");
      }
      positions(static_cast<Eigen::Index>(i), 0) = static_cast<DT>(xyz[0]);
      positions(static_cast<Eigen::Index>(i), 1) = static_cast<DT>(xyz[1]);
      positions(static_cast<Eigen::Index>(i), 2) = static_cast<DT>(xyz[2]);
    } else {
      float xyz[3];
      in.read(reinterpret_cast<char *>(xyz), sizeof(xyz));
      if (!in) {
        throw std::runtime_error("Failed to read TrxStream positions");
      }
      positions(static_cast<Eigen::Index>(i), 0) = static_cast<DT>(xyz[0]);
      positions(static_cast<Eigen::Index>(i), 1) = static_cast<DT>(xyz[1]);
      positions(static_cast<Eigen::Index>(i), 2) = static_cast<DT>(xyz[2]);
    }
  }

  for (const auto &kv : dps_) {
    trx.add_dps_from_vector(kv.first, kv.second.dtype, kv.second.values);
  }
  for (const auto &kv : dpv_) {
    trx.add_dpv_from_vector(kv.first, kv.second.dtype, kv.second.values);
  }
  for (const auto &kv : groups_) {
    trx.add_group_from_indices(kv.first, kv.second);
  }

  if (metadata_mode_ == MetadataMode::OnDisk) {
    for (const auto &meta : metadata_files_) {
      const std::string dest = trx._uncompressed_folder_handle + SEPARATOR + meta.relative_path;
      const trx::fs::path dest_path(dest);
      if (dest_path.has_parent_path()) {
        std::error_code parent_ec;
        trx::fs::create_directories(dest_path.parent_path(), parent_ec);
      }
      std::error_code copy_ec;
      trx::fs::copy_file(meta.absolute_path, dest, trx::fs::copy_options::overwrite_existing, copy_ec);
      if (copy_ec) {
        throw std::runtime_error("Failed to copy metadata file: " + meta.absolute_path + " -> " + dest);
      }
    }
  }

  trx.save(filename, compression_standard);
  trx.close();

  cleanup_tmp();
}

inline void TrxStream::finalize(const std::string &filename,
                                TrxScalarType output_dtype,
                                zip_uint32_t compression_standard) {
  switch (output_dtype) {
  case TrxScalarType::Float16:
    finalize<half>(filename, compression_standard);
    break;
  case TrxScalarType::Float64:
    finalize<double>(filename, compression_standard);
    break;
  case TrxScalarType::Float32:
  default:
    finalize<float>(filename, compression_standard);
    break;
  }
}

inline void TrxStream::finalize(const std::string &filename, const TrxSaveOptions &options) {
  if (options.mode == TrxSaveMode::Directory) {
    if (finalized_) {
      throw std::runtime_error("TrxStream already finalized");
    }
    if (options.overwrite_existing) {
      finalize_directory(filename);
    } else {
      finalize_directory_persistent(filename);
    }
    return;
  }

  TrxScalarType out_type = TrxScalarType::Float32;
  if (positions_dtype_ == "float16") {
    out_type = TrxScalarType::Float16;
  } else if (positions_dtype_ == "float64") {
    out_type = TrxScalarType::Float64;
  }
  finalize(filename, out_type, options.compression_standard);
}

inline void TrxStream::finalize_directory_impl(const std::string &directory, bool remove_existing) {
  if (finalized_) {
    throw std::runtime_error("TrxStream already finalized");
  }
  finalized_ = true;

  flush_positions_buffer();
  if (positions_out_.is_open()) {
    positions_out_.flush();
    positions_out_.close();
  }

  const size_t nb_streamlines = lengths_.size();
  const size_t nb_vertices = total_vertices_;

  std::error_code ec;
  if (remove_existing && trx::fs::exists(directory, ec)) {
    trx::fs::remove_all(directory, ec);
    ec.clear();
  }
  
  // Create directory if it doesn't exist
  if (!trx::fs::exists(directory, ec)) {
    trx::fs::create_directories(directory, ec);
    if (ec) {
      throw std::runtime_error("Failed to create output directory: " + directory);
    }
  }
  ec.clear();

  json header_out = header;
  header_out = _json_set(header_out, "NB_VERTICES", static_cast<int>(nb_vertices));
  header_out = _json_set(header_out, "NB_STREAMLINES", static_cast<int>(nb_streamlines));
  const std::string header_path = directory + SEPARATOR + "header.json";
  std::ofstream out_header(header_path, std::ios::out | std::ios::trunc);
  if (!out_header.is_open()) {
    throw std::runtime_error("Failed to write header.json to: " + header_path);
  }
  out_header << header_out.dump() << std::endl;
  out_header.close();

  const std::string positions_name = "positions.3." + positions_dtype_;
  const std::string positions_dst = directory + SEPARATOR + positions_name;
  trx::fs::rename(positions_path_, positions_dst, ec);
  if (ec) {
    ec.clear();
    trx::fs::copy_file(positions_path_, positions_dst, trx::fs::copy_options::overwrite_existing, ec);
    if (ec) {
      throw std::runtime_error("Failed to copy positions file to: " + positions_dst);
    }
  }

  const std::string offsets_dst = directory + SEPARATOR + "offsets.uint64";
  std::ofstream offsets_out(offsets_dst, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!offsets_out.is_open()) {
    throw std::runtime_error("Failed to open offsets file for write: " + offsets_dst);
  }
  uint64_t offset = 0;
  offsets_out.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
  for (const auto length : lengths_) {
    offset += static_cast<uint64_t>(length);
    offsets_out.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
  }
  offsets_out.flush();
  offsets_out.close();

  auto write_field_values = [&](const std::string &path, const FieldValues &values) {
    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open metadata file: " + path);
    }
    const size_t count = values.values.size();
    if (values.dtype == "float16") {
      const size_t chunk = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(half));
      std::vector<half> tmp;
      tmp.reserve(chunk);
      size_t idx = 0;
      while (idx < count) {
        const size_t n = std::min(chunk, count - idx);
        tmp.clear();
        for (size_t i = 0; i < n; ++i) {
          tmp.push_back(static_cast<half>(values.values[idx + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(n * sizeof(half)));
        idx += n;
      }
    } else if (values.dtype == "float32") {
      const size_t chunk = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(float));
      std::vector<float> tmp;
      tmp.reserve(chunk);
      size_t idx = 0;
      while (idx < count) {
        const size_t n = std::min(chunk, count - idx);
        tmp.clear();
        for (size_t i = 0; i < n; ++i) {
          tmp.push_back(static_cast<float>(values.values[idx + i]));
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(n * sizeof(float)));
        idx += n;
      }
    } else if (values.dtype == "float64") {
      const size_t chunk = std::max<std::size_t>(1, metadata_buffer_max_bytes_ / sizeof(double));
      std::vector<double> tmp;
      tmp.reserve(chunk);
      size_t idx = 0;
      while (idx < count) {
        const size_t n = std::min(chunk, count - idx);
        tmp.clear();
        for (size_t i = 0; i < n; ++i) {
          tmp.push_back(values.values[idx + i]);
        }
        out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(n * sizeof(double)));
        idx += n;
      }
    } else {
      throw std::runtime_error("Unsupported metadata dtype: " + values.dtype);
    }
    out.close();
  };

  if (metadata_mode_ == MetadataMode::OnDisk) {
    for (const auto &meta : metadata_files_) {
      const std::string dest = directory + SEPARATOR + meta.relative_path;
      const trx::fs::path dest_path(dest);
      if (dest_path.has_parent_path()) {
        std::error_code parent_ec;
        trx::fs::create_directories(dest_path.parent_path(), parent_ec);
      }
      std::error_code copy_ec;
      trx::fs::copy_file(meta.absolute_path, dest, trx::fs::copy_options::overwrite_existing, copy_ec);
      if (copy_ec) {
        throw std::runtime_error("Failed to copy metadata file: " + meta.absolute_path + " -> " + dest);
      }
    }
  } else {
    if (!dps_.empty()) {
      trx::fs::create_directories(directory + SEPARATOR + "dps", ec);
      for (const auto &kv : dps_) {
        const std::string path = directory + SEPARATOR + "dps" + SEPARATOR + kv.first + "." + kv.second.dtype;
        write_field_values(path, kv.second);
      }
    }
    if (!dpv_.empty()) {
      trx::fs::create_directories(directory + SEPARATOR + "dpv", ec);
      for (const auto &kv : dpv_) {
        const std::string path = directory + SEPARATOR + "dpv" + SEPARATOR + kv.first + "." + kv.second.dtype;
        write_field_values(path, kv.second);
      }
    }
    if (!groups_.empty()) {
      trx::fs::create_directories(directory + SEPARATOR + "groups", ec);
      for (const auto &kv : groups_) {
        const std::string path = directory + SEPARATOR + "groups" + SEPARATOR + kv.first + ".uint32";
        std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
          throw std::runtime_error("Failed to open group file: " + path);
        }
        if (!kv.second.empty()) {
          out.write(reinterpret_cast<const char *>(kv.second.data()),
                    static_cast<std::streamsize>(kv.second.size() * sizeof(uint32_t)));
        }
        out.close();
      }
    }
  }

  cleanup_tmp();
}

inline void TrxStream::finalize_directory(const std::string &directory) {
  finalize_directory_impl(directory, true);
}

inline void TrxStream::finalize_directory_persistent(const std::string &directory) {
  finalize_directory_impl(directory, false);
}

template <typename DT>
void TrxFile<DT>::add_dpv_from_tsf(const std::string &name, const std::string &dtype, const std::string &path) {
  if (name.empty()) {
    throw std::invalid_argument("DPV name cannot be empty");
  }

  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (!trx::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported DPV dtype: " + dtype);
  }
  if (dtype_norm != "float16" && dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported DPV dtype for TSF input: " + dtype);
  }

  if (!this->streamlines) {
    throw std::runtime_error("TRX file has no streamlines to attach DPV data");
  }
  if (this->_uncompressed_folder_handle.empty()) {
    throw std::runtime_error("TRX file has no backing directory to store DPV data");
  }

  const auto &lengths = this->streamlines->_lengths;
  const size_t nb_streamlines = static_cast<size_t>(lengths.size());
  const size_t nb_vertices = static_cast<size_t>(this->streamlines->_data.rows());

  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open TSF file: " + path);
  }

  auto trim = [](std::string note) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    note.erase(note.begin(),
               std::find_if(note.begin(), note.end(), [is_space](unsigned char ch) { return !is_space(ch); }));
    note.erase(std::find_if(note.rbegin(), note.rend(), [is_space](unsigned char ch) { return !is_space(ch); }).base(),
               note.end());
    return note;
  };

  std::streampos start_pos = input.tellg();
  std::string line;
  bool binary_mode = false;
  size_t data_offset = 0;
  std::string datatype;
  if (std::getline(input, line)) {
    const std::string first_line = trim(line);
    if (first_line == "mrtrix track scalars") {
      bool found_end = false;
      while (std::getline(input, line)) {
        const std::string trimmed = trim(line);
        if (trimmed == "END") {
          found_end = true;
          break;
        }
        const auto pos = trimmed.find(':');
        if (pos == std::string::npos) {
          continue;
        }
        const std::string key = trim(trimmed.substr(0, pos));
        const std::string value = trim(trimmed.substr(pos + 1));
        if (key == "datatype") {
          datatype = value;
        } else if (key == "file") {
          std::istringstream iss(value);
          std::string dot;
          iss >> dot >> data_offset;
          if (!iss.fail()) {
            binary_mode = true;
          }
        }
      }
      if (!found_end) {
        throw std::runtime_error("Failed to parse TSF header: missing END");
      }
    } else {
      input.clear();
      input.seekg(start_pos);
    }
  } else {
    throw std::runtime_error("Failed to parse TSF file: " + path);
  }

  std::vector<double> values;
  values.reserve(nb_vertices);
  size_t streamline_index = 0;
  uint32_t expected_vertices = nb_streamlines > 0 ? lengths(0) : 0;
  uint32_t current_vertices = 0;

  if (binary_mode) {
    if (datatype != "Float32LE" && datatype != "Float32BE" && datatype != "Float64LE" && datatype != "Float64BE") {
      throw std::runtime_error("Unsupported TSF datatype: " + datatype);
    }

    auto is_little_endian = []() {
      const uint16_t value = 1;
      return *reinterpret_cast<const uint8_t *>(&value) == 1;
    };
    const bool little_endian = is_little_endian();
    const bool data_little_endian = datatype.find("LE") != std::string::npos;

    input.clear();
    input.seekg(static_cast<std::streamoff>(data_offset));
    while (input.good()) {
      double value = 0.0;
      if (datatype == "Float32LE" || datatype == "Float32BE") {
        uint32_t raw = 0;
        input.read(reinterpret_cast<char *>(&raw), sizeof(raw));
        if (!input) {
          break;
        }
        if (little_endian != data_little_endian) {
          raw = (raw >> 24) | ((raw >> 8) & 0x0000FF00) | ((raw << 8) & 0x00FF0000) | (raw << 24);
        }
        float v = 0.0f;
        std::memcpy(&v, &raw, sizeof(v));
        value = static_cast<double>(v);
      } else {
        uint64_t raw = 0;
        input.read(reinterpret_cast<char *>(&raw), sizeof(raw));
        if (!input) {
          break;
        }
        if (little_endian != data_little_endian) {
          raw = ((raw & 0x00000000000000FFULL) << 56) | ((raw & 0x000000000000FF00ULL) << 40) |
                ((raw & 0x0000000000FF0000ULL) << 24) | ((raw & 0x00000000FF000000ULL) << 8) |
                ((raw & 0x000000FF00000000ULL) >> 8) | ((raw & 0x0000FF0000000000ULL) >> 24) |
                ((raw & 0x00FF000000000000ULL) >> 40) | ((raw & 0xFF00000000000000ULL) >> 56);
        }
        double v = 0.0;
        std::memcpy(&v, &raw, sizeof(v));
        value = v;
      }

      if (std::isinf(value)) {
        break;
      }
      if (std::isnan(value)) {
        if (current_vertices != expected_vertices) {
          throw std::runtime_error("TSF streamline length does not match TRX streamlines");
        }
        if (streamline_index + 1 < nb_streamlines) {
          ++streamline_index;
          expected_vertices = lengths(streamline_index);
          current_vertices = 0;
        }
        continue;
      }
      values.push_back(value);
      ++current_vertices;
    }
  } else {
    std::string token;
    while (input >> token) {
      std::string token_norm = token;
      std::transform(token_norm.begin(), token_norm.end(), token_norm.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (token_norm.rfind("nan", 0) == 0) {
        if (current_vertices != expected_vertices) {
          throw std::runtime_error("TSF streamline length does not match TRX streamlines");
        }
        if (streamline_index + 1 < nb_streamlines) {
          ++streamline_index;
          expected_vertices = lengths(streamline_index);
          current_vertices = 0;
        }
        continue;
      }
      if (token_norm.rfind("inf", 0) == 0) {
        break;
      }
      double value = 0.0;
      try {
        size_t idx = 0;
        value = std::stod(token, &idx);
        if (idx != token.size()) {
          throw std::invalid_argument("invalid token");
        }
      } catch (const std::exception &) {
        throw std::runtime_error("Failed to parse TSF file: " + path);
      }
      if (std::isinf(value)) {
        break;
      }
      if (std::isnan(value)) {
        if (current_vertices != expected_vertices) {
          throw std::runtime_error("TSF streamline length does not match TRX streamlines");
        }
        if (streamline_index + 1 < nb_streamlines) {
          ++streamline_index;
          expected_vertices = lengths(streamline_index);
          current_vertices = 0;
        }
        continue;
      }
      values.push_back(value);
      ++current_vertices;
    }
    if (!input.eof() && input.fail()) {
      throw std::runtime_error("Failed to parse TSF file: " + path);
    }
  }
  if (nb_streamlines > 0) {
    if (streamline_index != nb_streamlines - 1 || current_vertices != expected_vertices) {
      throw std::runtime_error("TSF streamline count does not match TRX streamlines");
    }
  }
  if (values.size() != nb_vertices) {
    throw std::runtime_error("TSF values (" + std::to_string(values.size()) + ") do not match number of vertices (" +
                             std::to_string(nb_vertices) + ")");
  }

  std::string dpv_dirname = this->_uncompressed_folder_handle + SEPARATOR + "dpv" + SEPARATOR;
  {
    std::error_code ec;
    trx::fs::create_directories(dpv_dirname, ec);
    if (ec) {
      throw std::runtime_error("Could not create directory " + dpv_dirname);
    }
  }

  std::string dpv_filename = dpv_dirname + name + "." + dtype_norm;
  {
    std::error_code ec;
    if (trx::fs::exists(dpv_filename, ec)) {
      trx::fs::remove(dpv_filename, ec);
    }
  }

  auto existing = this->data_per_vertex.find(name);
  if (existing != this->data_per_vertex.end()) {
    this->data_per_vertex.erase(existing);
  }

  const int rows = static_cast<int>(nb_vertices);
  const int cols = 1;
  std::tuple<int, int> shape = std::make_tuple(rows, cols);

  auto seq = std::make_unique<trx::ArraySequence<DT>>();
  seq->mmap_pos = trx::_create_memmap(dpv_filename, shape, "w+", dtype_norm);

  if (dtype_norm == "float16") {
    auto *data = reinterpret_cast<half *>(seq->mmap_pos.data());
    Map<Matrix<half, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(seq->_data)) Map<Matrix<half, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<half>(values[static_cast<size_t>(i)]);
    }
  } else if (dtype_norm == "float32") {
    auto *data = reinterpret_cast<float *>(seq->mmap_pos.data());
    Map<Matrix<float, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(seq->_data)) Map<Matrix<float, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<float>(values[static_cast<size_t>(i)]);
    }
  } else {
    auto *data = reinterpret_cast<double *>(seq->mmap_pos.data());
    Map<Matrix<double, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(seq->_data)) Map<Matrix<double, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows; ++i) {
      mapped(i, 0) = static_cast<double>(values[static_cast<size_t>(i)]);
    }
  }

  new (&(seq->_offsets)) Map<Matrix<uint64_t, Dynamic, Dynamic>>(this->streamlines->_offsets.data(),
                                                                 static_cast<int>(this->streamlines->_offsets.rows()),
                                                                 static_cast<int>(this->streamlines->_offsets.cols()));
  seq->_lengths = this->streamlines->_lengths;

  this->data_per_vertex[name] = std::move(seq);
}

template <typename DT>
void TrxFile<DT>::export_dpv_to_tsf(const std::string &name,
                                    const std::string &path,
                                    const std::string &timestamp,
                                    const std::string &dtype) const {
  if (name.empty()) {
    throw std::invalid_argument("DPV name cannot be empty");
  }
  if (timestamp.empty()) {
    throw std::invalid_argument("TSF timestamp cannot be empty");
  }

  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (!trx::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported TSF dtype: " + dtype);
  }
  if (dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported TSF dtype for output: " + dtype);
  }

  if (!this->streamlines) {
    throw std::runtime_error("TRX file has no streamlines to export DPV data");
  }

  const auto dpv_it = this->data_per_vertex.find(name);
  if (dpv_it == this->data_per_vertex.end()) {
    throw std::runtime_error("DPV entry not found: " + name);
  }

  const auto *seq = dpv_it->second.get();
  if (!seq) {
    throw std::runtime_error("DPV entry is null: " + name);
  }
  if (seq->_data.cols() != 1) {
    throw std::runtime_error("DPV must be 1D to export as TSF: " + name);
  }

  const auto &lengths = this->streamlines->_lengths;
  const size_t nb_streamlines = static_cast<size_t>(lengths.size());
  const size_t nb_vertices = static_cast<size_t>(seq->_data.rows());
  if (nb_vertices != static_cast<size_t>(this->streamlines->_data.rows())) {
    throw std::runtime_error("DPV vertex count does not match streamlines data");
  }

  const auto is_little_endian = []() {
    const uint16_t value = 1;
    return *reinterpret_cast<const uint8_t *>(&value) == 1;
  };
  const bool little_endian = is_little_endian();
  const std::string dtype_spec = dtype_norm == "float64" ? (little_endian ? "Float64LE" : "Float64BE")
                                                         : (little_endian ? "Float32LE" : "Float32BE");

  auto build_header = [&](size_t data_offset) {
    std::ostringstream header;
    header << "mrtrix track scalars\n";
    header << "timestamp: " << timestamp << "\n";
    header << "datatype: " << dtype_spec << "\n";
    header << "file: . " << data_offset << "\n";
    header << "count: " << nb_streamlines << "\n";
    header << "total_count: " << nb_streamlines << "\n";
    header << "END\n";
    return header.str();
  };

  size_t data_offset = 0;
  for (int i = 0; i < 4; ++i) {
    const std::string header = build_header(data_offset);
    size_t padded = header.size();
    const size_t pad = (4 - (padded % 4)) % 4;
    padded += pad;
    if (padded == data_offset) {
      break;
    }
    data_offset = padded;
  }
  const std::string header = build_header(data_offset);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open TSF file for writing: " + path);
  }
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  const size_t pad = (4 - (header.size() % 4)) % 4;
  if (pad > 0) {
    const std::array<char, 4> zeros{0, 0, 0, 0};
    out.write(zeros.data(), static_cast<std::streamsize>(pad));
  }

  const auto write_value = [&](double value) {
    if (dtype_norm == "float64") {
      const double cast = value;
      out.write(reinterpret_cast<const char *>(&cast), sizeof(cast));
    } else {
      const float cast = static_cast<float>(value);
      out.write(reinterpret_cast<const char *>(&cast), sizeof(cast));
    }
  };

  const size_t total_vertices = static_cast<size_t>(seq->_data.rows());
  size_t offset = 0;
  for (size_t s = 0; s < nb_streamlines; ++s) {
    const uint32_t len = lengths(static_cast<Eigen::Index>(s));
    if (offset > total_vertices) {
      throw std::runtime_error("DPV length metadata exceeds vertex count");
    }
    if (len > std::numeric_limits<size_t>::max() - offset) {
      throw std::runtime_error("DPV length metadata exceeds vertex count");
    }
    if (offset + static_cast<size_t>(len) > total_vertices) {
      throw std::runtime_error("DPV length metadata exceeds vertex count");
    }
    offset += static_cast<size_t>(len);
  }
  offset = 0;
  for (size_t s = 0; s < nb_streamlines; ++s) {
    const uint32_t len = lengths(static_cast<Eigen::Index>(s));
    for (uint32_t i = 0; i < len; ++i) {
      const size_t idx = offset + static_cast<size_t>(i);
      if (idx > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max())) {
        throw std::runtime_error("DPV length metadata exceeds vertex count");
      }
      write_value(static_cast<double>(seq->_data(static_cast<Eigen::Index>(idx), 0)));
    }
    offset += static_cast<size_t>(len);
    if (s + 1 < nb_streamlines) {
      write_value(std::numeric_limits<double>::quiet_NaN());
    }
  }
  write_value(std::numeric_limits<double>::infinity());

  if (!out.good()) {
    throw std::runtime_error("Failed to write TSF file: " + path);
  }
}

template <typename DT> std::ostream &operator<<(std::ostream &out, const TrxFile<DT> &trx) {

  out << "Header (header.json):\n";
  out << trx.header.dump();
  return out;
}

template <typename DT>
std::vector<std::array<Eigen::half, 6>> TrxFile<DT>::build_streamline_aabbs() const {
  std::vector<std::array<Eigen::half, 6>> aabbs;
  if (!this->streamlines) {
    return aabbs;
  }

  std::vector<uint64_t> offsets;
  if (this->streamlines->_offsets.size() > 0) {
    offsets.resize(static_cast<size_t>(this->streamlines->_offsets.size()));
    for (Eigen::Index i = 0; i < this->streamlines->_offsets.size(); ++i) {
      offsets[static_cast<size_t>(i)] = this->streamlines->_offsets(i, 0);
    }
  } else if (this->streamlines->_lengths.size() > 0) {
    const size_t nb_streamlines = static_cast<size_t>(this->streamlines->_lengths.size());
    offsets.resize(nb_streamlines + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < nb_streamlines; ++i) {
      offsets[i + 1] = offsets[i] + static_cast<uint64_t>(this->streamlines->_lengths(static_cast<Eigen::Index>(i)));
    }
  } else {
    return aabbs;
  }

  const size_t nb_streamlines = offsets.size() > 0 ? offsets.size() - 1 : 0;
  aabbs.resize(nb_streamlines);

  for (size_t i = 0; i < nb_streamlines; ++i) {
    const uint64_t start = offsets[i];
    const uint64_t end = offsets[i + 1];
    if (end <= start) {
      aabbs[i] = {Eigen::half(0), Eigen::half(0), Eigen::half(0),
                  Eigen::half(0), Eigen::half(0), Eigen::half(0)};
      continue;
    }

    float min_x = std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float min_z = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();

    for (uint64_t p = start; p < end; ++p) {
      const float x = static_cast<float>(this->streamlines->_data(static_cast<Eigen::Index>(p), 0));
      const float y = static_cast<float>(this->streamlines->_data(static_cast<Eigen::Index>(p), 1));
      const float z = static_cast<float>(this->streamlines->_data(static_cast<Eigen::Index>(p), 2));
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      min_z = std::min(min_z, z);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);
      max_z = std::max(max_z, z);
    }

    aabbs[i] = {static_cast<Eigen::half>(min_x), static_cast<Eigen::half>(min_y), static_cast<Eigen::half>(min_z),
                static_cast<Eigen::half>(max_x), static_cast<Eigen::half>(max_y), static_cast<Eigen::half>(max_z)};
  }

  this->aabb_cache_ = aabbs;
  return aabbs;
}

template <typename DT>
const std::vector<std::array<Eigen::half, 6>> &TrxFile<DT>::get_or_build_streamline_aabbs() const {
  if (this->aabb_cache_.empty()) {
    this->build_streamline_aabbs();
  }
  return this->aabb_cache_;
}

template <typename DT>
std::unique_ptr<TrxFile<DT>> TrxFile<DT>::query_aabb(
    const std::array<float, 3> &min_corner,
    const std::array<float, 3> &max_corner,
    const std::vector<std::array<Eigen::half, 6>> *precomputed_aabbs,
    bool build_cache_for_result) const {
  if (!this->streamlines) {
    auto empty = std::make_unique<TrxFile<DT>>();
    empty->header = _json_set(this->header, "NB_VERTICES", 0);
    empty->header = _json_set(empty->header, "NB_STREAMLINES", 0);
    return empty;
  }

  size_t nb_streamlines = 0;
  if (this->streamlines->_offsets.size() > 0) {
    nb_streamlines = static_cast<size_t>(this->streamlines->_offsets.size() - 1);
  } else if (this->streamlines->_lengths.size() > 0) {
    nb_streamlines = static_cast<size_t>(this->streamlines->_lengths.size());
  } else {
    auto empty = std::make_unique<TrxFile<DT>>();
    empty->header = _json_set(this->header, "NB_VERTICES", 0);
    empty->header = _json_set(empty->header, "NB_STREAMLINES", 0);
    return empty;
  }

  std::vector<std::array<Eigen::half, 6>> aabbs_local;
  const std::vector<std::array<Eigen::half, 6>> &aabbs = precomputed_aabbs
      ? *precomputed_aabbs
      : (!this->aabb_cache_.empty() ? this->aabb_cache_ : (aabbs_local = this->build_streamline_aabbs()));
  if (aabbs.size() != nb_streamlines) {
    throw std::invalid_argument("AABB size does not match streamlines count");
  }

  const float min_x = min_corner[0];
  const float min_y = min_corner[1];
  const float min_z = min_corner[2];
  const float max_x = max_corner[0];
  const float max_y = max_corner[1];
  const float max_z = max_corner[2];

  std::vector<uint32_t> selected;
  selected.reserve(nb_streamlines);

  for (size_t i = 0; i < nb_streamlines; ++i) {
    const auto &box = aabbs[i];
    const float box_min_x = static_cast<float>(box[0]);
    const float box_min_y = static_cast<float>(box[1]);
    const float box_min_z = static_cast<float>(box[2]);
    const float box_max_x = static_cast<float>(box[3]);
    const float box_max_y = static_cast<float>(box[4]);
    const float box_max_z = static_cast<float>(box[5]);

    if (box_min_x <= max_x && box_max_x >= min_x &&
        box_min_y <= max_y && box_max_y >= min_y &&
        box_min_z <= max_z && box_max_z >= min_z) {
      selected.push_back(static_cast<uint32_t>(i));
    }
  }

  return this->subset_streamlines(selected, build_cache_for_result);
}

template <typename DT>
void TrxFile<DT>::invalidate_aabb_cache() const {
  this->aabb_cache_.clear();
}

template <typename DT>
const MMappedMatrix<DT> *TrxFile<DT>::get_dps(const std::string &name) const {
  auto it = this->data_per_streamline.find(name);
  if (it == this->data_per_streamline.end()) {
    return nullptr;
  }
  return it->second.get();
}

template <typename DT>
const ArraySequence<DT> *TrxFile<DT>::get_dpv(const std::string &name) const {
  auto it = this->data_per_vertex.find(name);
  if (it == this->data_per_vertex.end()) {
    return nullptr;
  }
  return it->second.get();
}

template <typename DT>
std::vector<std::array<DT, 3>> TrxFile<DT>::get_streamline(size_t streamline_index) const {
  if (!this->streamlines || this->streamlines->_offsets.size() == 0) {
    throw std::runtime_error("TRX streamlines are not available");
  }
  const size_t n_streamlines = static_cast<size_t>(this->streamlines->_offsets.size() - 1);
  if (streamline_index >= n_streamlines) {
    throw std::out_of_range("Streamline index out of range");
  }

  const uint64_t start = static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(streamline_index), 0));
  const uint64_t end =
      static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(streamline_index + 1), 0));
  std::vector<std::array<DT, 3>> points;
  if (end <= start) {
    return points;
  }
  points.reserve(static_cast<size_t>(end - start));
  for (uint64_t i = start; i < end; ++i) {
    points.push_back({this->streamlines->_data(static_cast<Eigen::Index>(i), 0),
                      this->streamlines->_data(static_cast<Eigen::Index>(i), 1),
                      this->streamlines->_data(static_cast<Eigen::Index>(i), 2)});
  }
  return points;
}

template <typename DT>
template <typename Fn>
void TrxFile<DT>::for_each_streamline(Fn &&fn) const {
  if (!this->streamlines || this->streamlines->_offsets.size() == 0) {
    return;
  }
  const size_t n_streamlines = static_cast<size_t>(this->streamlines->_offsets.size() - 1);
  for (size_t i = 0; i < n_streamlines; ++i) {
    const uint64_t start = static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(i), 0));
    const uint64_t end = static_cast<uint64_t>(this->streamlines->_offsets(static_cast<Eigen::Index>(i + 1), 0));
    fn(i, start, end - start);
  }
}

template <typename DT>
template <typename T>
void TrxFile<DT>::add_dpg_from_vector(const std::string &group,
                                      const std::string &name,
                                      const std::string &dtype,
                                      const std::vector<T> &values,
                                      int rows,
                                      int cols) {
  if (group.empty()) {
    throw std::invalid_argument("DPG group cannot be empty");
  }
  if (name.empty()) {
    throw std::invalid_argument("DPG name cannot be empty");
  }
  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (!trx::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported DPG dtype: " + dtype);
  }
  if (dtype_norm != "float16" && dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported DPG dtype: " + dtype);
  }
  if (this->_uncompressed_folder_handle.empty()) {
    throw std::runtime_error("TRX file has no backing directory to store DPG data");
  }
  if (rows <= 0) {
    throw std::invalid_argument("DPG rows must be positive");
  }
  if (cols < 0) {
    if (values.size() % static_cast<size_t>(rows) != 0) {
      throw std::invalid_argument("DPG values size does not match rows");
    }
    cols = static_cast<int>(values.size() / static_cast<size_t>(rows));
  }
  if (cols <= 0) {
    throw std::invalid_argument("DPG cols must be positive");
  }
  if (static_cast<size_t>(rows) * static_cast<size_t>(cols) != values.size()) {
    throw std::invalid_argument("DPG values size does not match rows*cols");
  }

  std::string dpg_dir = this->_uncompressed_folder_handle + SEPARATOR + "dpg" + SEPARATOR;
  {
    std::error_code ec;
    trx::fs::create_directories(dpg_dir, ec);
    if (ec) {
      throw std::runtime_error("Could not create directory " + dpg_dir);
    }
  }
  std::string dpg_subdir = dpg_dir + group;
  {
    std::error_code ec;
    trx::fs::create_directories(dpg_subdir, ec);
    if (ec) {
      throw std::runtime_error("Could not create directory " + dpg_subdir);
    }
  }

  std::string dpg_filename = dpg_subdir + SEPARATOR + name + "." + dtype_norm;
  {
    std::error_code ec;
    if (trx::fs::exists(dpg_filename, ec)) {
      trx::fs::remove(dpg_filename, ec);
    }
  }

  auto &group_map = this->data_per_group[group];
  group_map.erase(name);

  std::tuple<int, int> shape = std::make_tuple(rows, cols);
  group_map[name] = std::make_unique<MMappedMatrix<DT>>();
  group_map[name]->mmap = _create_memmap(dpg_filename, shape, "w+", dtype_norm);

  if (dtype_norm == "float16") {
    auto *data = reinterpret_cast<half *>(group_map[name]->mmap.data());
    Map<Matrix<half, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(group_map[name]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = static_cast<half>(values[static_cast<size_t>(i)]);
    }
  } else if (dtype_norm == "float32") {
    auto *data = reinterpret_cast<float *>(group_map[name]->mmap.data());
    Map<Matrix<float, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(group_map[name]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = static_cast<float>(values[static_cast<size_t>(i)]);
    }
  } else {
    auto *data = reinterpret_cast<double *>(group_map[name]->mmap.data());
    Map<Matrix<double, Dynamic, Dynamic>> mapped(data, rows, cols);
    new (&(group_map[name]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(data, rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
      data[i] = static_cast<double>(values[static_cast<size_t>(i)]);
    }
  }
}

template <typename DT>
template <typename Derived>
void TrxFile<DT>::add_dpg_from_matrix(const std::string &group,
                                      const std::string &name,
                                      const std::string &dtype,
                                      const Eigen::MatrixBase<Derived> &matrix) {
  if (matrix.size() == 0) {
    throw std::invalid_argument("DPG matrix cannot be empty");
  }
  std::vector<typename Derived::Scalar> values;
  values.reserve(static_cast<size_t>(matrix.size()));
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      values.push_back(matrix(i, j));
    }
  }
  add_dpg_from_vector(group, name, dtype, values, static_cast<int>(matrix.rows()),
                      static_cast<int>(matrix.cols()));
}

template <typename DT>
const MMappedMatrix<DT> *TrxFile<DT>::get_dpg(const std::string &group, const std::string &name) const {
  auto group_it = this->data_per_group.find(group);
  if (group_it == this->data_per_group.end()) {
    return nullptr;
  }
  auto field_it = group_it->second.find(name);
  if (field_it == group_it->second.end()) {
    return nullptr;
  }
  return field_it->second.get();
}

template <typename DT>
std::vector<std::string> TrxFile<DT>::list_dpg_groups() const {
  std::vector<std::string> groups;
  groups.reserve(this->data_per_group.size());
  for (const auto &kv : this->data_per_group) {
    groups.push_back(kv.first);
  }
  return groups;
}

template <typename DT>
std::vector<std::string> TrxFile<DT>::list_dpg_fields(const std::string &group) const {
  std::vector<std::string> fields;
  auto it = this->data_per_group.find(group);
  if (it == this->data_per_group.end()) {
    return fields;
  }
  fields.reserve(it->second.size());
  for (const auto &kv : it->second) {
    fields.push_back(kv.first);
  }
  return fields;
}

template <typename DT>
void TrxFile<DT>::remove_dpg(const std::string &group, const std::string &name) {
  auto group_it = this->data_per_group.find(group);
  if (group_it == this->data_per_group.end()) {
    return;
  }
  group_it->second.erase(name);
  if (group_it->second.empty()) {
    this->data_per_group.erase(group_it);
  }
}

template <typename DT>
void TrxFile<DT>::remove_dpg_group(const std::string &group) {
  this->data_per_group.erase(group);
}

template <typename DT>
std::unique_ptr<TrxFile<DT>> TrxFile<DT>::subset_streamlines(const std::vector<uint32_t> &streamline_ids,
                                                             bool build_cache_for_result) const {
  if (!this->streamlines) {
    auto empty = std::make_unique<TrxFile<DT>>();
    empty->header = _json_set(this->header, "NB_VERTICES", 0);
    empty->header = _json_set(empty->header, "NB_STREAMLINES", 0);
    return empty;
  }

  std::vector<uint64_t> offsets;
  if (this->streamlines->_offsets.size() > 0) {
    offsets.resize(static_cast<size_t>(this->streamlines->_offsets.size()));
    for (Eigen::Index i = 0; i < this->streamlines->_offsets.size(); ++i) {
      offsets[static_cast<size_t>(i)] = this->streamlines->_offsets(i, 0);
    }
  } else if (this->streamlines->_lengths.size() > 0) {
    const size_t nb_streamlines = static_cast<size_t>(this->streamlines->_lengths.size());
    offsets.resize(nb_streamlines + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < nb_streamlines; ++i) {
      offsets[i + 1] = offsets[i] + static_cast<uint64_t>(this->streamlines->_lengths(static_cast<Eigen::Index>(i)));
    }
  } else {
    auto empty = std::make_unique<TrxFile<DT>>();
    empty->header = _json_set(this->header, "NB_VERTICES", 0);
    empty->header = _json_set(empty->header, "NB_STREAMLINES", 0);
    return empty;
  }

  const size_t nb_streamlines = offsets.size() > 0 ? offsets.size() - 1 : 0;
  if (nb_streamlines == 0) {
    auto empty = std::make_unique<TrxFile<DT>>();
    empty->header = _json_set(this->header, "NB_VERTICES", 0);
    empty->header = _json_set(empty->header, "NB_STREAMLINES", 0);
    return empty;
  }

  std::vector<uint32_t> selected;
  selected.reserve(streamline_ids.size());
  std::vector<uint8_t> seen(nb_streamlines, 0);
  for (uint32_t id : streamline_ids) {
    if (id >= nb_streamlines) {
      throw std::invalid_argument("Streamline id out of range");
    }
    if (!seen[id]) {
      selected.push_back(id);
      seen[id] = 1;
    }
  }

  if (selected.empty()) {
    auto empty = std::make_unique<TrxFile<DT>>();
    empty->header = _json_set(this->header, "NB_VERTICES", 0);
    empty->header = _json_set(empty->header, "NB_STREAMLINES", 0);
    return empty;
  }

  std::vector<int> old_to_new(nb_streamlines, -1);
  size_t total_vertices = 0;
  for (size_t i = 0; i < selected.size(); ++i) {
    const uint32_t idx = selected[i];
    old_to_new[idx] = static_cast<int>(i);
    const uint64_t start = offsets[idx];
    const uint64_t end = offsets[idx + 1];
    total_vertices += static_cast<size_t>(end - start);
  }

  auto out = std::make_unique<TrxFile<DT>>(static_cast<int>(total_vertices),
                                           static_cast<int>(selected.size()),
                                           this);
  out->header = _json_set(this->header, "NB_VERTICES", static_cast<int>(total_vertices));
  out->header = _json_set(out->header, "NB_STREAMLINES", static_cast<int>(selected.size()));

  auto &out_positions = out->streamlines->_data;
  auto &out_offsets = out->streamlines->_offsets;
  auto &out_lengths = out->streamlines->_lengths;

  size_t cursor = 0;
  out_offsets(0, 0) = 0;
  for (size_t new_idx = 0; new_idx < selected.size(); ++new_idx) {
    const uint32_t old_idx = selected[new_idx];
    const uint64_t start = offsets[old_idx];
    const uint64_t end = offsets[old_idx + 1];
    const uint64_t len = end - start;

    out_lengths(static_cast<Eigen::Index>(new_idx)) = static_cast<uint32_t>(len);
    out_offsets(static_cast<Eigen::Index>(new_idx + 1), 0) =
        out_offsets(static_cast<Eigen::Index>(new_idx), 0) + len;

    if (len > 0) {
      out_positions.block(static_cast<Eigen::Index>(cursor), 0,
                          static_cast<Eigen::Index>(len), 3) =
          this->streamlines->_data.block(static_cast<Eigen::Index>(start), 0,
                                         static_cast<Eigen::Index>(len), 3);

      for (const auto &kv : this->data_per_vertex) {
        const std::string &name = kv.first;
        auto out_it = out->data_per_vertex.find(name);
        if (out_it == out->data_per_vertex.end()) {
          continue;
        }
        auto &out_dpv = out_it->second->_data;
        auto &src_dpv = kv.second->_data;
        const Eigen::Index cols = src_dpv.cols();
        out_dpv.block(static_cast<Eigen::Index>(cursor), 0,
                      static_cast<Eigen::Index>(len), cols) =
            src_dpv.block(static_cast<Eigen::Index>(start), 0,
                          static_cast<Eigen::Index>(len), cols);
      }
    }

    for (const auto &kv : this->data_per_streamline) {
      const std::string &name = kv.first;
      auto out_it = out->data_per_streamline.find(name);
      if (out_it == out->data_per_streamline.end()) {
        continue;
      }
      out_it->second->_matrix.row(static_cast<Eigen::Index>(new_idx)) =
          kv.second->_matrix.row(static_cast<Eigen::Index>(old_idx));
    }

    cursor += static_cast<size_t>(len);
  }

  for (const auto &kv : this->groups) {
    const std::string &group_name = kv.first;
    std::vector<uint32_t> indices;
    auto &matrix = kv.second->_matrix;
    indices.reserve(static_cast<size_t>(matrix.size()));
    for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
      for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
        const uint32_t old_idx = matrix(r, c);
        if (old_idx >= old_to_new.size()) {
          continue;
        }
        const int new_idx = old_to_new[old_idx];
        if (new_idx >= 0) {
          indices.push_back(static_cast<uint32_t>(new_idx));
        }
      }
    }
    if (!indices.empty()) {
      out->add_group_from_indices(group_name, indices);
    }
  }

  if (!this->data_per_group.empty() && !out->groups.empty()) {
    std::string dpg_dir = out->_uncompressed_folder_handle + SEPARATOR + "dpg" + SEPARATOR;
    {
      std::error_code ec;
      trx::fs::create_directories(dpg_dir, ec);
      if (ec) {
        throw std::runtime_error("Could not create directory " + dpg_dir);
      }
    }

    for (const auto &group_kv : out->groups) {
      const std::string &group_name = group_kv.first;
      auto src_group_it = this->data_per_group.find(group_name);
      if (src_group_it == this->data_per_group.end()) {
        continue;
      }

      std::string dpg_subdir = dpg_dir + group_name;
      {
        std::error_code ec;
        trx::fs::create_directories(dpg_subdir, ec);
        if (ec) {
          throw std::runtime_error("Could not create directory " + dpg_subdir);
        }
      }

      if (out->data_per_group.find(group_name) == out->data_per_group.end()) {
        out->data_per_group.emplace(group_name, std::map<std::string, std::unique_ptr<MMappedMatrix<DT>>>{});
      } else {
        out->data_per_group[group_name].clear();
      }

      for (const auto &field_kv : src_group_it->second) {
        const std::string &field_name = field_kv.first;
        std::string dpg_dtype = dtype_from_scalar<DT>();
        std::string dpg_filename = dpg_subdir + SEPARATOR + field_name;
        dpg_filename = _generate_filename_from_data(field_kv.second->_matrix, dpg_filename);

        std::tuple<int, int> dpg_shape = std::make_tuple(field_kv.second->_matrix.rows(),
                                                         field_kv.second->_matrix.cols());

        out->data_per_group[group_name][field_name] = std::make_unique<MMappedMatrix<DT>>();
        out->data_per_group[group_name][field_name]->mmap =
            _create_memmap(dpg_filename, dpg_shape, "w+", dpg_dtype);

        if (dpg_dtype.compare("float16") == 0) {
          new (&(out->data_per_group[group_name][field_name]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(
              reinterpret_cast<half *>(out->data_per_group[group_name][field_name]->mmap.data()),
              std::get<0>(dpg_shape), std::get<1>(dpg_shape));
        } else if (dpg_dtype.compare("float32") == 0) {
          new (&(out->data_per_group[group_name][field_name]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(
              reinterpret_cast<float *>(out->data_per_group[group_name][field_name]->mmap.data()),
              std::get<0>(dpg_shape), std::get<1>(dpg_shape));
        } else {
          new (&(out->data_per_group[group_name][field_name]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(
              reinterpret_cast<double *>(out->data_per_group[group_name][field_name]->mmap.data()),
              std::get<0>(dpg_shape), std::get<1>(dpg_shape));
        }

        for (int i = 0; i < out->data_per_group[group_name][field_name]->_matrix.rows(); ++i) {
          for (int j = 0; j < out->data_per_group[group_name][field_name]->_matrix.cols(); ++j) {
            out->data_per_group[group_name][field_name]->_matrix(i, j) =
                field_kv.second->_matrix(i, j);
          }
        }
      }
    }
  }

  if (build_cache_for_result) {
    out->build_streamline_aabbs();
  }
  return out;
}

#ifdef TRX_TPP_OPEN_NAMESPACE
} // namespace trx
#undef TRX_TPP_OPEN_NAMESPACE
#endif
