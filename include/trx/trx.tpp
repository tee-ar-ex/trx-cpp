// Taken from: https://stackoverflow.com/a/25389481
#ifndef TRX_H
#define TRX_TPP_STANDALONE
#define TRX_TPP_OPEN_NAMESPACE
#include <trx/trx.h>
#undef TRX_TPP_STANDALONE
namespace trxmmap {
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
    positions_dtype = dtype_from_scalar<half>();
    offsets_dtype = dtype_from_scalar<uint64_t>();
    lengths_dtype = dtype_from_scalar<uint32_t>();
  }
  std::string positions_filename(tmp_dir);
  positions_filename += "/positions.3." + positions_dtype;

  std::tuple<int, int> shape = std::make_tuple(nb_vertices, 3);

  trx->streamlines = std::make_unique<ArraySequence<DT>>();
  trx->streamlines->mmap_pos = trxmmap::_create_memmap(positions_filename, shape, "w+", positions_dtype);

  // TODO: find a better way to get the dtype than using all these switch cases. Also refactor
  // into function as per specifications, positions can only be floats
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

  trx->streamlines->mmap_off = trxmmap::_create_memmap(offsets_filename, shape_off, "w+", offsets_dtype);
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
      trx->data_per_vertex[x.first]->mmap_pos = trxmmap::_create_memmap(dpv_filename, dpv_shape, "w+", dpv_dtype);
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
      trx->data_per_streamline[x.first] = std::make_unique<trxmmap::MMappedMatrix<DT>>();
      trx->data_per_streamline[x.first]->mmap =
          trxmmap::_create_memmap(dps_filename, dps_shape, std::string("w+"), dps_dtype);

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
  auto trx = std::make_unique<trxmmap::TrxFile<DT>>();
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
    std::tuple<std::string, int, std::string> base_tuple =
        trxmmap::detail::_split_ext_with_dimensionality(elem_filename);
    std::string base(std::get<0>(base_tuple));
    int dim = std::get<1>(base_tuple);
    std::string ext(std::get<2>(base_tuple));

    if (ext.compare("bit") == 0) {
      ext = "bool";
    }

    long long mem_adress = std::get<0>(x->second);
    long long size = std::get<1>(x->second);

    if (base.compare("positions") == 0 && (folder.compare("") == 0 || folder.compare(".") == 0)) {
      if (size != static_cast<int>(trx->header["NB_VERTICES"].int_value()) * 3 || dim != 3) {

        throw std::invalid_argument("Wrong data size/dimensionality");
      }

      std::tuple<int, int> shape = std::make_tuple(static_cast<int>(trx->header["NB_VERTICES"].int_value()), 3);
      trx->streamlines->mmap_pos =
          trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

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
      if (size != static_cast<int>(trx->header["NB_STREAMLINES"].int_value()) + 1 || dim != 1) {
        throw std::invalid_argument(
            "Wrong offsets size/dimensionality: size=" + std::to_string(size) +
            " nb_streamlines=" + std::to_string(static_cast<int>(trx->header["NB_STREAMLINES"].int_value())) +
            " dim=" + std::to_string(dim) + " filename=" + elem_filename);
      }

      const int nb_str = static_cast<int>(trx->header["NB_STREAMLINES"].int_value());
      std::tuple<int, int> shape = std::make_tuple(nb_str + 1, 1);
      trx->streamlines->mmap_off = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

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
          trxmmap::detail::_compute_lengths(offsets, static_cast<int>(trx->header["NB_VERTICES"].int_value()));
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
      trx->data_per_streamline[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

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
      trx->data_per_vertex[base]->mmap_pos = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

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
      trx->data_per_group[sub_folder][data_name]->mmap =
          trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

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
      trx->groups[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);
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
    auto empty_copy = std::make_unique<trxmmap::TrxFile<DT>>();
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

  int last_elem_pos = trxmmap::detail::_dichotomic_search(this->streamlines->_lengths);

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
      trx->groups[x.first]->mmap = trxmmap::_create_memmap(group_name, group_shape, "w+", group_dtype);
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
  std::ifstream header_file(header_name);
  if (!header_file.is_open()) {
    throw std::runtime_error("Failed to open header.json at: " + header_name);
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

  return TrxFile<DT>::_create_trx_from_pointer(header, files_pointer_size, "", directory);
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
  std::string ext = get_ext(filename);

  if (ext.size() > 0 && (ext != "zip" && ext != "trx")) {
    throw std::invalid_argument("Unsupported extension." + ext);
  }

  auto copy_trx = this->deepcopy();
  copy_trx->resize();
  std::string tmp_dir_name = copy_trx->_uncompressed_folder_handle;

  if (ext.size() > 0 && (ext == "zip" || ext == "trx")) {
    int errorp;
    zip_t *zf;
    if ((zf = zip_open(filename.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp)) == nullptr) {
      throw std::runtime_error("Could not open archive " + filename + ": " + strerror(errorp));
    } else {
      zip_from_folder(zf, tmp_dir_name, tmp_dir_name, compression_standard);
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
    copy_trx->close();
  }
}

template <typename DT>
void TrxFile<DT>::add_dps_from_text(const std::string &name, const std::string &dtype, const std::string &path) {
  if (name.empty()) {
    throw std::invalid_argument("DPS name cannot be empty");
  }

  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (!trxmmap::detail::_is_dtype_valid(dtype_norm)) {
    throw std::invalid_argument("Unsupported DPS dtype: " + dtype);
  }
  if (dtype_norm != "float16" && dtype_norm != "float32" && dtype_norm != "float64") {
    throw std::invalid_argument("Unsupported DPS dtype for text input: " + dtype);
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

  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open DPS text file: " + path);
  }

  std::vector<double> values;
  values.reserve(nb_streamlines);
  double value = 0.0;
  while (input >> value) {
    values.push_back(value);
  }
  if (!input.eof() && input.fail()) {
    throw std::runtime_error("Failed to parse DPS text file: " + path);
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

  auto matrix = std::make_unique<trxmmap::MMappedMatrix<DT>>();
  matrix->mmap = trxmmap::_create_memmap(dps_filename, shape, "w+", dtype_norm);

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
void TrxFile<DT>::add_dpv_from_tsf(const std::string &name, const std::string &dtype, const std::string &path) {
  if (name.empty()) {
    throw std::invalid_argument("DPV name cannot be empty");
  }

  std::string dtype_norm = dtype;
  std::transform(dtype_norm.begin(), dtype_norm.end(), dtype_norm.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (!trxmmap::detail::_is_dtype_valid(dtype_norm)) {
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

  auto seq = std::make_unique<trxmmap::ArraySequence<DT>>();
  seq->mmap_pos = trxmmap::_create_memmap(dpv_filename, shape, "w+", dtype_norm);

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

  if (!trxmmap::detail::_is_dtype_valid(dtype_norm)) {
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

#ifdef TRX_TPP_OPEN_NAMESPACE
} // namespace trxmmap
#undef TRX_TPP_OPEN_NAMESPACE
#endif
