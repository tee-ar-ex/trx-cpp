// Taken from: https://stackoverflow.com/a/25389481
template <class Matrix>
void write_binary(const char *filename, const Matrix &matrix)
{
	std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
	// out.write((char *)(&rows), sizeof(typename Matrix::Index));
	// out.write((char *)(&cols), sizeof(typename Matrix::Index));
	out.write((char *)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
	out.close();
}
template <class Matrix>
void read_binary(const char *filename, Matrix &matrix)
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	typename Matrix::Index rows = 0, cols = 0;
	in.read((char *)(&rows), sizeof(typename Matrix::Index));
	in.read((char *)(&cols), sizeof(typename Matrix::Index));
	matrix.resize(rows, cols);
	in.read((char *)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
	in.close();
}

template <typename DT>
void ediff1d(Matrix<DT, Dynamic, 1> &lengths, Matrix<DT, Dynamic, Dynamic> &tmp, uint32_t to_end)
{
	Map<RowVector<uint32_t, Dynamic>> v(tmp.data(), tmp.size());
	lengths.resize(v.size(), 1);

	// TODO: figure out if there's a built in way to manage this
	for (int i = 0; i < v.size() - 1; i++)
	{
		lengths(i) = v(i + 1) - v(i);
	}
	lengths(v.size() - 1) = to_end;
}

template <typename DT>
// Caveat: if filename has an extension, it will be replaced by the generated dtype extension.
std::string _generate_filename_from_data(const MatrixBase<DT> &arr, std::string filename)
{

	std::string base, ext;

	base = filename; // get_base(SEPARATOR, filename);
	ext = get_ext(filename);

	if (ext.size() != 0)
	{
		base = base.substr(0, base.length() - ext.length() - 1);
	}

	std::string dt = dtype_from_scalar<typename DT::Scalar>();

	Eigen::Index n_cols = arr.cols();

	std::string new_filename;
	if (n_cols == 1)
	{
		new_filename = base + "." + dt;
	}
	else
	{
		new_filename = base + "." + std::to_string(static_cast<long long>(n_cols)) + "." + dt;
	}

	return new_filename;
}

template <typename DT>
Matrix<uint32_t, Dynamic, 1> _compute_lengths(const MatrixBase<DT> &offsets, int nb_vertices)
{
	if (offsets.size() > 1)
	{
		const auto casted = offsets.template cast<uint64_t>();
		const Eigen::Index len = offsets.size() - 1;
		Matrix<uint32_t, Dynamic, 1> lengths(len);
		for (Eigen::Index i = 0; i < len; ++i)
		{
			lengths(i) = static_cast<uint32_t>(casted(i + 1) - casted(i));
		}
		return lengths;
	}
	// If offsets are empty or only contain the sentinel, there are zero streamlines.
	return Matrix<uint32_t, Dynamic, 1>(0);
}

template <typename DT>
int _dichotomic_search(const MatrixBase<DT> &x, int l_bound, int r_bound)
{
	if (l_bound == -1 && r_bound == -1)
	{
		l_bound = 0;
		r_bound = static_cast<int>(x.size()) - 1;
	}

	if (l_bound == r_bound)
	{
		int val;
		if (x(l_bound) != 0)
			val = l_bound;
		else
			val = -1;
		return val;
	}

	int mid_bound = (l_bound + r_bound + 1) / 2;

	if (x(mid_bound) == 0)
		return _dichotomic_search(x, l_bound, mid_bound - 1);
	else
		return _dichotomic_search(x, mid_bound, r_bound);
}

template <typename DT>
TrxFile<DT>::TrxFile(int nb_vertices, int nb_streamlines, const TrxFile<DT> *init_as, std::string reference)
{
	std::vector<std::vector<float>> affine(4);
	std::vector<uint16_t> dimensions(3);

	// TODO: check if there's a more efficient way to do this with Eigen
	if (init_as != NULL)
	{
		for (int i = 0; i < 4; i++)
		{
			affine[i] = {0, 0, 0, 0};
			for (int j = 0; j < 4; j++)
			{
				affine[i][j] = float(init_as->header["VOXEL_TO_RASMM"][i][j].number_value());
			}
		}

		for (int i = 0; i < 3; i++)
		{
			dimensions[i] = uint16_t(init_as->header["DIMENSIONS"][i].int_value());
		}
	}
	// TODO: add else if for get_reference_info
	else
	{
		// identity matrix
		for (int i = 0; i < 4; i++)
		{
			affine[i] = {0, 0, 0, 0};
			affine[i][i] = 1;
		}
		dimensions = {1, 1, 1};
	}

	if (nb_vertices == 0 && nb_streamlines == 0)
	{
		if (init_as != NULL)
		{
			// raise error here
			throw std::invalid_argument("Can't us init_as without declaring nb_vertices and nb_streamlines");
		}

		// will remove as completely unecessary. using as placeholders
		this->header = {};
		this->streamlines = nullptr;

		// TODO: maybe create a matrix to map to of specified DT. Do we need this??
		// set default datatype to half
		// default data is null so will not set data. User will need configure desired datatype
		// this->streamlines = ArraySequence<half>();
		this->_uncompressed_folder_handle = "";

		nb_vertices = 0;
		nb_streamlines = 0;
	}
	else if (nb_vertices > 0 && nb_streamlines > 0)
	{
		TrxFile<DT> *trx = _initialize_empty_trx<DT>(nb_streamlines, nb_vertices, init_as);
		this->streamlines = trx->streamlines;
		this->groups = trx->groups;
		this->data_per_streamline = trx->data_per_streamline;
		this->data_per_vertex = trx->data_per_vertex;
		this->data_per_group = trx->data_per_group;
		this->_uncompressed_folder_handle = trx->_uncompressed_folder_handle;
		this->_owns_uncompressed_folder = trx->_owns_uncompressed_folder;
		this->_copy_safe = trx->_copy_safe;
		trx->_owns_uncompressed_folder = false;
		trx->_uncompressed_folder_handle.clear();
		delete trx;
	}
	else
	{
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
TrxFile<DT> *_initialize_empty_trx(int nb_streamlines, int nb_vertices, const TrxFile<DT> *init_as)
{
	TrxFile<DT> *trx = new TrxFile<DT>();

	std::string tmp_dir = make_temp_dir("trx");

	json header = json::object();
	if (init_as != NULL)
	{
		header = init_as->header;
	}
	header = _json_set(header, "NB_VERTICES", nb_vertices);
	header = _json_set(header, "NB_STREAMLINES", nb_streamlines);

	std::string positions_dtype;
	std::string offsets_dtype;
	std::string lengths_dtype;

	if (init_as != NULL)
	{
		header = _json_set(header, "VOXEL_TO_RASMM", init_as->header["VOXEL_TO_RASMM"]);
		header = _json_set(header, "DIMENSIONS", init_as->header["DIMENSIONS"]);
		positions_dtype = dtype_from_scalar<DT>();
		offsets_dtype = dtype_from_scalar<uint64_t>();
		lengths_dtype = dtype_from_scalar<uint32_t>();
	}
	else
	{
		positions_dtype = dtype_from_scalar<half>();
		offsets_dtype = dtype_from_scalar<uint64_t>();
		lengths_dtype = dtype_from_scalar<uint32_t>();
	}
	std::string positions_filename(tmp_dir);
	positions_filename += "/positions.3." + positions_dtype;

	std::tuple<int, int> shape = std::make_tuple(nb_vertices, 3);

	trx->streamlines = new ArraySequence<DT>();
	trx->streamlines->mmap_pos = trxmmap::_create_memmap(positions_filename, shape, "w+", positions_dtype);

	// TODO: find a better way to get the dtype than using all these switch cases. Also refactor into function
	// as per specifications, positions can only be floats
	if (positions_dtype.compare("float16") == 0)
	{
		new (&(trx->streamlines->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
	}
	else if (positions_dtype.compare("float32") == 0)
	{
		new (&(trx->streamlines->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
	}
	else
	{
		new (&(trx->streamlines->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
	}

	std::string offsets_filename(tmp_dir);
	offsets_filename += "/offsets." + offsets_dtype;

	std::tuple<int, int> shape_off = std::make_tuple(nb_streamlines + 1, 1);

	trx->streamlines->mmap_off = trxmmap::_create_memmap(offsets_filename, shape_off, "w+", offsets_dtype);
	new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape_off), std::get<1>(shape_off));

	trx->streamlines->_lengths.resize(nb_streamlines);
	trx->streamlines->_lengths.setZero();

	if (init_as != NULL)
	{
		std::string dpv_dirname;
		std::string dps_dirname;
		if (init_as->data_per_vertex.size() > 0)
		{
			dpv_dirname = tmp_dir + "/dpv/";
			mkdir(dpv_dirname.c_str(), S_IRWXU);
		}
		if (init_as->data_per_streamline.size() > 0)
		{
			dps_dirname = tmp_dir + "/dps/";
			mkdir(dps_dirname.c_str(), S_IRWXU);
		}

		for (auto const &x : init_as->data_per_vertex)
		{
			int rows, cols;
			std::string dpv_dtype = dtype_from_scalar<DT>();
			Map<Matrix<DT, Dynamic, Dynamic, RowMajor>> tmp_as = init_as->data_per_vertex.find(x.first)->second->_data;

			std::string dpv_filename;
			if (tmp_as.rows() == 1)
			{
				dpv_filename = dpv_dirname + x.first + "." + dpv_dtype;
				rows = nb_vertices;
				cols = 1;
			}
			else
			{
				rows = nb_vertices;
				cols = tmp_as.cols();

				dpv_filename = dpv_dirname + x.first + "." + std::to_string(cols) + "." + dpv_dtype;
			}

			std::tuple<int, int> dpv_shape = std::make_tuple(rows, cols);
			trx->data_per_vertex[x.first] = new ArraySequence<DT>();
			trx->data_per_vertex[x.first]->mmap_pos = trxmmap::_create_memmap(dpv_filename, dpv_shape, "w+", dpv_dtype);
			if (dpv_dtype.compare("float16") == 0)
			{
				new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
			}
			else if (dpv_dtype.compare("float32") == 0)
			{
				new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
			}
			else
			{
				new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
			}

			new (&(trx->data_per_vertex[x.first]->_offsets)) Map<Matrix<uint64_t, Dynamic, Dynamic>>(trx->streamlines->_offsets.data(), int(trx->streamlines->_offsets.rows()), int(trx->streamlines->_offsets.cols()));
			trx->data_per_vertex[x.first]->_lengths = trx->streamlines->_lengths;
		}

		for (auto const &x : init_as->data_per_streamline)
		{
			std::string dps_dtype = dtype_from_scalar<DT>();
			int rows, cols;
			Map<Matrix<DT, Dynamic, Dynamic>> tmp_as = init_as->data_per_streamline.find(x.first)->second->_matrix;

			std::string dps_filename;

			if (tmp_as.rows() == 1)
			{
				dps_filename = dps_dirname + x.first + "." + dps_dtype;
				rows = nb_streamlines;
			}
			else
			{
				cols = tmp_as.cols();
				rows = nb_streamlines;

				dps_filename = dps_dirname + x.first + "." + std::to_string(cols) + "." + dps_dtype;
			}

			std::tuple<int, int> dps_shape = std::make_tuple(rows, cols);
			trx->data_per_streamline[x.first] = new trxmmap::MMappedMatrix<DT>();
			trx->data_per_streamline[x.first]->mmap = trxmmap::_create_memmap(dps_filename, dps_shape, std::string("w+"), dps_dtype);

			if (dps_dtype.compare("float16") == 0)
			{
				new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
			}
			else if (dps_dtype.compare("float32") == 0)
			{
				new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
			}
			else
			{
				new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
			}
		}
	}

	trx->header = header;
	trx->_uncompressed_folder_handle = tmp_dir;
	trx->_owns_uncompressed_folder = true;

	return trx;
}

template <typename DT>
TrxFile<DT> *TrxFile<DT>::_create_trx_from_pointer(json header, std::map<std::string, std::tuple<long long, long long>> dict_pointer_size, std::string root_zip, std::string root)
{
	trxmmap::TrxFile<DT> *trx = new trxmmap::TrxFile<DT>();
	trx->header = header;
	trx->streamlines = new ArraySequence<DT>();

	std::string filename;

	// TODO: Fix this hack of iterating through dictionary in reverse to get main files read first
	for (auto x = dict_pointer_size.rbegin(); x != dict_pointer_size.rend(); ++x)
	{
		std::string elem_filename = x->first;

		if (root_zip.size() > 0)
		{
			filename = root_zip;
		}
		else
		{
			filename = elem_filename;
		}

		std::string folder = path_dirname(elem_filename);

		// _split_ext_with_dimensionality
		std::tuple<std::string, int, std::string> base_tuple = _split_ext_with_dimensionality(elem_filename);
		std::string base(std::get<0>(base_tuple));
		int dim = std::get<1>(base_tuple);
		std::string ext(std::get<2>(base_tuple));

		if (ext.compare("bit") == 0)
		{
			ext = "bool";
		}

		long long mem_adress = std::get<0>(x->second);
		long long size = std::get<1>(x->second);

		std::string stripped = root;

		// TODO : will not work on windows
		if (stripped.rfind("/") == stripped.size() - 1)
		{
			stripped = stripped.substr(0, stripped.size() - 1);
		}

		if (root.compare("") != 0 && folder.rfind(stripped, stripped.size()) == 0)
		{
			// 1 for the first forward slash
			folder = folder.replace(0, root.size(), "");

			if (folder[0] == SEPARATOR.c_str()[0])
			{
				folder = folder.substr(1, folder.size());
			}
		}

		if (base.compare("positions") == 0 && (folder.compare("") == 0 || folder.compare(".") == 0))
		{
		if (size != static_cast<int>(trx->header["NB_VERTICES"].int_value()) * 3 || dim != 3)
			{

				throw std::invalid_argument("Wrong data size/dimensionality");
			}

		std::tuple<int, int> shape = std::make_tuple(static_cast<int>(trx->header["NB_VERTICES"].int_value()), 3);
			trx->streamlines->mmap_pos = trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

			// TODO: find a better way to get the dtype than using all these switch cases. Also refactor into function
			// as per specifications, positions can only be floats
			if (ext.compare("float16") == 0)
			{
				new (&(trx->streamlines->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("float32") == 0)
			{
				new (&(trx->streamlines->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->streamlines->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
		}

		else if (base.compare("offsets") == 0 && (folder.compare("") == 0 || folder.compare(".") == 0))
		{
		if (size != static_cast<int>(trx->header["NB_STREAMLINES"].int_value()) + 1 || dim != 1)
			{
				throw std::invalid_argument("Wrong offsets size/dimensionality: size=" +
							    std::to_string(size) + " nb_streamlines=" +
						    std::to_string(static_cast<int>(trx->header["NB_STREAMLINES"].int_value())) +
							    " dim=" + std::to_string(dim) + " filename=" + elem_filename);
			}

		const int nb_str = static_cast<int>(trx->header["NB_STREAMLINES"].int_value());
			std::tuple<int, int> shape = std::make_tuple(nb_str + 1, 1);
			trx->streamlines->mmap_off = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

			if (ext.compare("uint64") == 0)
			{
				new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("uint32") == 0)
			{
				trx->streamlines->_offsets_owned.resize(std::get<0>(shape));
				auto *src = reinterpret_cast<const uint32_t *>(trx->streamlines->mmap_off.data());
				for (int i = 0; i < std::get<0>(shape); ++i)
					trx->streamlines->_offsets_owned[size_t(i)] = uint64_t(src[i]);
				new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(trx->streamlines->_offsets_owned.data(), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				throw std::invalid_argument("Unsupported offsets datatype: " + ext);
			}

			Matrix<uint64_t, Dynamic, 1> offsets = trx->streamlines->_offsets;
		trx->streamlines->_lengths = _compute_lengths(offsets, static_cast<int>(trx->header["NB_VERTICES"].int_value()));
		}

		else if (folder.compare("dps") == 0)
		{
			std::tuple<int, int> shape;
			trx->data_per_streamline[base] = new MMappedMatrix<DT>();
		int nb_scalar = size / static_cast<int>(trx->header["NB_STREAMLINES"].int_value());

		if (size % static_cast<int>(trx->header["NB_STREAMLINES"].int_value()) != 0 || nb_scalar != dim)
			{

				throw std::invalid_argument("Wrong dps size/dimensionality");
			}
			else
			{
			shape = std::make_tuple(static_cast<int>(trx->header["NB_STREAMLINES"].int_value()), nb_scalar);
			}
			trx->data_per_streamline[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

			if (ext.compare("float16") == 0)
			{
				new (&(trx->data_per_streamline[base]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_streamline[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("float32") == 0)
			{
				new (&(trx->data_per_streamline[base]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_streamline[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->data_per_streamline[base]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_streamline[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
		}

		else if (folder.compare("dpv") == 0)
		{
			std::tuple<int, int> shape;
			trx->data_per_vertex[base] = new ArraySequence<DT>();
		int nb_scalar = size / static_cast<int>(trx->header["NB_VERTICES"].int_value());

		if (size % static_cast<int>(trx->header["NB_VERTICES"].int_value()) != 0 || nb_scalar != dim)
			{

				throw std::invalid_argument("Wrong dpv size/dimensionality");
			}
			else
			{
			shape = std::make_tuple(static_cast<int>(trx->header["NB_VERTICES"].int_value()), nb_scalar);
			}
			trx->data_per_vertex[base]->mmap_pos = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

			if (ext.compare("float16") == 0)
			{
				new (&(trx->data_per_vertex[base]->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_vertex[base]->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("float32") == 0)
			{
				new (&(trx->data_per_vertex[base]->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_vertex[base]->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->data_per_vertex[base]->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_vertex[base]->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}

			new (&(trx->data_per_vertex[base]->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(trx->streamlines->_offsets.data(), std::get<0>(shape), std::get<1>(shape));
			trx->data_per_vertex[base]->_lengths = trx->streamlines->_lengths;
		}

		else if (folder.rfind("dpg", 0) == 0)
		{
			std::tuple<int, int> shape;

			if (size != dim)
			{

				throw std::invalid_argument("Wrong dpg size/dimensionality");
			}
			else
			{
				shape = std::make_tuple(1, size);
			}

			std::string data_name = path_basename(base);
			std::string sub_folder = path_basename(folder);

			trx->data_per_group[sub_folder][data_name] = new MMappedMatrix<DT>();
			trx->data_per_group[sub_folder][data_name]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

			if (ext.compare("float16") == 0)
			{
				new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_group[sub_folder][data_name]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("float32") == 0)
			{
				new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_group[sub_folder][data_name]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_group[sub_folder][data_name]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
		}

		else if (folder.compare("groups") == 0)
		{
			std::tuple<int, int> shape;
			if (dim != 1)
			{
				throw std::invalid_argument("Wrong group dimensionality");
			}
			else
			{
				shape = std::make_tuple(size, 1);
			}
			trx->groups[base] = new MMappedMatrix<uint32_t>();
			trx->groups[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);
			new (&(trx->groups[base]->_matrix)) Map<Matrix<uint32_t, Dynamic, Dynamic>>(reinterpret_cast<uint32_t *>(trx->groups[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
		}
		else
		{
			throw std::invalid_argument("Entry is not part of a valid TRX structure: " + elem_filename);
		}
	}
	if (trx->streamlines->_data.size() == 0 || trx->streamlines->_offsets.size() == 0)
	{

		throw std::invalid_argument("Missing essential data.");
	}

	return trx;
}

// TODO: Major refactoring
template <typename DT>
TrxFile<DT> *TrxFile<DT>::deepcopy()
{
	if (this->streamlines == nullptr || this->streamlines->_data.size() == 0 ||
	    this->streamlines->_offsets.size() == 0)
	{
		trxmmap::TrxFile<DT> *empty_copy = new trxmmap::TrxFile<DT>();
		empty_copy->header = this->header;
		return empty_copy;
	}
	std::string tmp_dir = make_temp_dir("trx");

	std::string header = tmp_dir + SEPARATOR + "header.json";
	std::ofstream out_json(header);

	// TODO: Definitely a better way to deepcopy
	json tmp_header = this->header;

	ArraySequence<DT> *to_dump = new ArraySequence<DT>();
	// TODO: Verify that this is indeed a deep copy
	new (&(to_dump->_data)) Matrix<DT, Dynamic, Dynamic, RowMajor>(this->streamlines->_data);
	new (&(to_dump->_offsets)) Matrix<uint64_t, Dynamic, Dynamic, RowMajor>(this->streamlines->_offsets);
	new (&(to_dump->_lengths)) Matrix<uint32_t, Dynamic, 1>(this->streamlines->_lengths);

	if (!this->_copy_safe)
	{
		const int nb_streamlines = to_dump->_offsets.size() > 0 ? static_cast<int>(to_dump->_offsets.size() - 1) : 0;
		const int nb_vertices = static_cast<int>(to_dump->_data.size() / 3);
		tmp_header = _json_set(tmp_header, "NB_STREAMLINES", nb_streamlines);
		tmp_header = _json_set(tmp_header, "NB_VERTICES", nb_vertices);
	}
	// Ensure sentinel is correct before persisting
	if (to_dump->_offsets.size() > 0)
	{
		to_dump->_offsets(to_dump->_offsets.size() - 1) = static_cast<uint64_t>(tmp_header["NB_VERTICES"].int_value());
	}
	if (out_json.is_open())
	{
		out_json << tmp_header.dump() << std::endl;
		out_json.close();
	}

	std::string pos_rootfn = tmp_dir + SEPARATOR + "positions";
	std::string positions_filename = _generate_filename_from_data(to_dump->_data, pos_rootfn);

	write_binary(positions_filename.c_str(), to_dump->_data);

	std::string off_rootfn = tmp_dir + SEPARATOR + "offsets";
	std::string offsets_filename = _generate_filename_from_data(to_dump->_offsets, off_rootfn);

	write_binary(offsets_filename.c_str(), to_dump->_offsets);

	if (this->data_per_vertex.size() > 0)
	{
		std::string dpv_dirname = tmp_dir + SEPARATOR + "dpv" + SEPARATOR;
		if (mkdir(dpv_dirname.c_str(), S_IRWXU) != 0)
		{
			throw std::runtime_error("Could not create directory " + dpv_dirname);
		}
		for (auto const &x : this->data_per_vertex)
		{
			Matrix<DT, Dynamic, Dynamic, RowMajor> dpv_todump = x.second->_data;
			std::string dpv_filename = dpv_dirname + x.first;
			dpv_filename = _generate_filename_from_data(dpv_todump, dpv_filename);

			write_binary(dpv_filename.c_str(), dpv_todump);
		}
	}

	if (this->data_per_streamline.size() > 0)
	{
		std::string dps_dirname = tmp_dir + SEPARATOR + "dps" + SEPARATOR;
		if (mkdir(dps_dirname.c_str(), S_IRWXU) != 0)
		{
			throw std::runtime_error("Could not create directory " + dps_dirname);
		}
		for (auto const &x : this->data_per_streamline)
		{
			Matrix<DT, Dynamic, Dynamic> dps_todump = x.second->_matrix;
			std::string dps_filename = dps_dirname + x.first;
			dps_filename = _generate_filename_from_data(dps_todump, dps_filename);

			write_binary(dps_filename.c_str(), dps_todump);
		}
	}

	if (this->groups.size() > 0)
	{
		std::string groups_dirname = tmp_dir + SEPARATOR + "groups" + SEPARATOR;
		if (mkdir(groups_dirname.c_str(), S_IRWXU) != 0)
		{
			throw std::runtime_error("Could not create directory " + groups_dirname);
		}

		for (auto const &x : this->groups)
		{
			Matrix<uint32_t, Dynamic, Dynamic> group_todump = x.second->_matrix;
			std::string group_filename = groups_dirname + x.first;
			group_filename = _generate_filename_from_data(group_todump, group_filename);

			write_binary(group_filename.c_str(), group_todump);

			if (this->data_per_group.find(x.first) == this->data_per_group.end())
			{
				continue;
			}

			for (auto const &y : this->data_per_group[x.first])
			{
				std::string dpg_dirname = tmp_dir + SEPARATOR + "dpg" + SEPARATOR;
				std::string dpg_subdirname = dpg_dirname + x.first;
				struct stat sb;

				if (stat(dpg_dirname.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
				{
					if (mkdir(dpg_dirname.c_str(), S_IRWXU) != 0)
					{
						throw std::runtime_error("Could not create directory " + dpg_dirname);
					}
				}
				if (stat(dpg_subdirname.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
				{
					if (mkdir(dpg_subdirname.c_str(), S_IRWXU) != 0)
					{
						throw std::runtime_error("Could not create directory " + dpg_subdirname);
					}
				}

				Matrix<DT, Dynamic, Dynamic> dpg_todump = this->data_per_group[x.first][y.first]->_matrix;
				std::string dpg_filename = dpg_subdirname + SEPARATOR + y.first;
				dpg_filename = _generate_filename_from_data(dpg_todump, dpg_filename);

				write_binary(dpg_filename.c_str(), dpg_todump);
			}
		}
	}

	TrxFile<DT> *copy_trx = load_from_directory<DT>(tmp_dir);
	copy_trx->_uncompressed_folder_handle = tmp_dir;
	copy_trx->_owns_uncompressed_folder = true;

	return copy_trx;
}

// TODO: verify that this function is actually necessary (there should not be preallocation zeros afaik)
template <typename DT>
std::tuple<int, int> TrxFile<DT>::_get_real_len()
{
	if (this->streamlines->_lengths.size() == 0)
		return std::make_tuple(0, 0);

	int last_elem_pos = _dichotomic_search(this->streamlines->_lengths);

	if (last_elem_pos != -1)
	{
		int strs_end = last_elem_pos + 1;
		int pts_end = this->streamlines->_lengths(seq(0, last_elem_pos), 0).sum();

		return std::make_tuple(strs_end, pts_end);
	}

	return std::make_tuple(0, 0);
}

template <typename DT>
std::tuple<int, int> TrxFile<DT>::_copy_fixed_arrays_from(TrxFile<DT> *trx, int strs_start, int pts_start, int nb_strs_to_copy)
{
	int curr_strs_len, curr_pts_len;

	if (nb_strs_to_copy == -1)
	{
		std::tuple<int, int> curr = this->_get_real_len();
		curr_strs_len = std::get<0>(curr);
		curr_pts_len = std::get<1>(curr);
	}
	else
	{
		curr_strs_len = nb_strs_to_copy;
		curr_pts_len = trx->streamlines->_lengths(seq(0, curr_strs_len - 1)).sum();
	}

	if (pts_start == -1)
	{
		pts_start = 0;
	}
	if (strs_start == -1)
	{
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

	for (auto const &x : this->data_per_vertex)
	{
		this->data_per_vertex[x.first]->_data.block(pts_start, 0, curr_pts_len, this->data_per_vertex[x.first]->_data.cols()) =
		    trx->data_per_vertex[x.first]->_data.block(0, 0, curr_pts_len, trx->data_per_vertex[x.first]->_data.cols());
		new (&(this->data_per_vertex[x.first]->_offsets)) Map<Matrix<uint64_t, Dynamic, Dynamic>>(trx->data_per_vertex[x.first]->_offsets.data(), trx->data_per_vertex[x.first]->_offsets.rows(), trx->data_per_vertex[x.first]->_offsets.cols());
		this->data_per_vertex[x.first]->_lengths = trx->data_per_vertex[x.first]->_lengths;
	}

	for (auto const &x : this->data_per_streamline)
	{
		this->data_per_streamline[x.first]->_matrix.block(strs_start, 0, curr_strs_len, this->data_per_streamline[x.first]->_matrix.cols()) =
		    trx->data_per_streamline[x.first]->_matrix.block(0, 0, curr_strs_len, trx->data_per_streamline[x.first]->_matrix.cols());
	}

	return std::make_tuple(strs_end, pts_end);
}

template <typename DT>
void TrxFile<DT>::close()
{
	this->_cleanup_temporary_directory();
	*this = TrxFile<DT>(); // probably dangerous to do
}

template <typename DT>
TrxFile<DT>::~TrxFile()
{
	this->_cleanup_temporary_directory();
}

template <typename DT>
// Caveat: cleanup is best-effort; filesystem errors are ignored.
void TrxFile<DT>::_cleanup_temporary_directory()
{
	if (this->_owns_uncompressed_folder && !this->_uncompressed_folder_handle.empty())
	{
		if (rm_dir(this->_uncompressed_folder_handle.c_str()) != 0)
		{
		}
		this->_uncompressed_folder_handle.clear();
		this->_owns_uncompressed_folder = false;
	}
}

template <typename DT>
// Caveats: downsizing vertices is not supported; reducing streamlines truncates data; same-size resize is a no-op.
void TrxFile<DT>::resize(int nb_streamlines, int nb_vertices, bool delete_dpg)
{
	if (!this->_copy_safe)
	{
		std::invalid_argument("Cannot resize a sliced dataset.");
	}

	std::tuple<int, int> sp_end = this->_get_real_len();
	int strs_end = std::get<0>(sp_end);
	int ptrs_end = std::get<1>(sp_end);

	if (nb_streamlines != -1 && nb_streamlines < strs_end)
	{
		strs_end = nb_streamlines;
	}

	if (nb_vertices == -1)
	{
		ptrs_end = this->streamlines->_lengths.sum();
		nb_vertices = ptrs_end;
	}
	else if (nb_vertices < ptrs_end)
	{
		return;
	}

	if (nb_streamlines == -1)
	{
		nb_streamlines = strs_end;
	}

	if (nb_streamlines == this->header["NB_STREAMLINES"].int_value() &&
	    nb_vertices == this->header["NB_VERTICES"].int_value())
	{
		return;
	}

	TrxFile<DT> *trx = _initialize_empty_trx(nb_streamlines, nb_vertices, this);

	if (nb_streamlines < this->header["NB_STREAMLINES"].int_value())
		trx->_copy_fixed_arrays_from(this, -1, -1, nb_streamlines);
	else
	{
		trx->_copy_fixed_arrays_from(this);
	}

	std::string tmp_dir = trx->_uncompressed_folder_handle;

	if (this->groups.size() > 0)
	{
		std::string group_dir = tmp_dir + SEPARATOR + "groups" + SEPARATOR;
		if (mkdir(group_dir.c_str(), S_IRWXU) != 0)
		{
			throw std::runtime_error("Could not create directory " + group_dir);
		}

		for (auto const &x : this->groups)
		{
			std::string group_dtype = dtype_from_scalar<uint32_t>();
			std::string group_name = group_dir + x.first + "." + group_dtype;

			int ori_length = this->groups[x.first]->_matrix.size();

			std::vector<int> keep_rows;
			std::vector<int> keep_cols = {0};

			// Slicing
			for (int i = 0; i < x.second->_matrix.rows(); ++i)
			{
				for (int j = 0; j < x.second->_matrix.cols(); ++j)
				{
					if (static_cast<int>(x.second->_matrix(i, j)) < strs_end)
					{
						keep_rows.push_back(i);
					}
				}
			}
			// std::cout << "Cols " << keep_rows.at(1) << std::endl;

			Matrix<uint32_t, Dynamic, Dynamic> tmp = this->groups[x.first]->_matrix(keep_rows, keep_cols);
			std::tuple<int, int> group_shape = std::make_tuple(tmp.size(), 1);

			trx->groups[x.first] = new MMappedMatrix<uint32_t>();
			trx->groups[x.first]->mmap = trxmmap::_create_memmap(group_name, group_shape, "w+", group_dtype);
			new (&(trx->groups[x.first]->_matrix)) Map<Matrix<uint32_t, Dynamic, Dynamic>>(reinterpret_cast<uint32_t *>(trx->groups[x.first]->mmap.data()), std::get<0>(group_shape), std::get<1>(group_shape));

			// update values
			for (int i = 0; i < trx->groups[x.first]->_matrix.rows(); ++i)
			{
				for (int j = 0; j < trx->groups[x.first]->_matrix.cols(); ++j)
				{
					trx->groups[x.first]->_matrix(i, j) = tmp(i, j);
				}
			}
		}
	}

	if (delete_dpg)
	{
		this->close();
		return;
	}

	if (this->data_per_group.size() > 0)
	{
		// really need to refactor all these mkdirs
		std::string dpg_dir = tmp_dir + SEPARATOR + "dpg" + SEPARATOR;
		if (mkdir(dpg_dir.c_str(), S_IRWXU) != 0)
		{
			throw std::runtime_error("Could not create directory " + dpg_dir);
		}

		for (auto const &x : this->data_per_group)
		{
			std::string dpg_subdir = dpg_dir + x.first;
			struct stat sb;

			if (stat(dpg_subdir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
			{
				if (mkdir(dpg_subdir.c_str(), S_IRWXU) != 0)
				{
					throw std::runtime_error("Could not create directory " + dpg_subdir);
				}
			}

			if (trx->data_per_group.find(x.first) == trx->data_per_group.end())
			{
				trx->data_per_group[x.first] = {};
			}

			for (auto const &y : this->data_per_group[x.first])
			{
				std::string dpg_dtype = dtype_from_scalar<DT>();
				std::string dpg_filename = dpg_subdir + SEPARATOR + y.first;
				dpg_filename = _generate_filename_from_data(this->data_per_group[x.first][y.first]->_matrix, dpg_filename);

				std::tuple<int, int> dpg_shape = std::make_tuple(this->data_per_group[x.first][y.first]->_matrix.rows(), this->data_per_group[x.first][y.first]->_matrix.cols());

				if (trx->data_per_group[x.first].find(y.first) == trx->data_per_group[x.first].end())
				{
					trx->data_per_group[x.first][y.first] = new MMappedMatrix<DT>();
				}

				trx->data_per_group[x.first][y.first]->mmap = _create_memmap(dpg_filename, dpg_shape, "w+", dpg_dtype);
				new (&(trx->data_per_group[x.first][y.first]->_matrix)) Map<Matrix<uint32_t, Dynamic, Dynamic>>(reinterpret_cast<uint32_t *>(trx->data_per_group[x.first][y.first]->mmap.data()), std::get<0>(dpg_shape), std::get<1>(dpg_shape));

				// update values
				for (int i = 0; i < trx->data_per_group[x.first][y.first]->_matrix.rows(); ++i)
				{
					for (int j = 0; j < trx->data_per_group[x.first][y.first]->_matrix.cols(); ++j)
					{
						trx->data_per_group[x.first][y.first]->_matrix(i, j) = this->data_per_group[x.first][y.first]->_matrix(i, j);
					}
				}
			}
		}
		this->close();
	}
}

template <typename DT>
TrxFile<DT> *load_from_zip(std::string filename)
{
	int errorp = 0;
	zip_t *zf = zip_open(filename.c_str(), 0, &errorp);
	if (zf == nullptr)
	{
		throw std::runtime_error("Could not open zip file: " + filename);
	}

	std::string temp_dir = extract_zip_to_directory(zf);
	zip_close(zf);

	TrxFile<DT> *trx = load_from_directory<DT>(temp_dir);
	trx->_uncompressed_folder_handle = temp_dir;
	trx->_owns_uncompressed_folder = true;
	return trx;
}

template <typename DT>
TrxFile<DT> *load_from_directory(std::string path)
{
	std::string directory = path;
	char resolved[PATH_MAX];
	if (realpath(path.c_str(), resolved) != nullptr)
	{
		directory = resolved;
	}
	std::string header_name = directory + SEPARATOR + "header.json";

	// TODO: add check to verify that it's open
	std::ifstream header_file(header_name);
	if (!header_file.is_open())
	{
		throw std::runtime_error("Failed to open header.json at: " + header_name);
	}
	std::string jstream((std::istreambuf_iterator<char>(header_file)),
			    std::istreambuf_iterator<char>());
	header_file.close();
	std::string err;
	json header = json::parse(jstream, err);
	if (!err.empty())
	{
		throw std::runtime_error("Failed to parse header.json: " + err);
	}

	std::map<std::string, std::tuple<long long, long long>> files_pointer_size;
	populate_fps(directory.c_str(), files_pointer_size);

	return TrxFile<DT>::_create_trx_from_pointer(header, files_pointer_size, "", directory);
}

template <typename DT>
TrxFile<DT> *load(std::string path)
{
	trx::fs::path input(path);
	if (!trx::fs::exists(input))
	{
		throw std::runtime_error("Input path does not exist: " + path);
	}
	std::error_code ec;
	if (trx::fs::is_directory(input, ec) && !ec)
	{
		return load_from_directory<DT>(path);
	}
	return load_from_zip<DT>(path);
}

template <typename DT>
TrxReader<DT>::TrxReader(const std::string &path)
{
	trx_ = load<DT>(path);
}

template <typename DT>
TrxReader<DT>::~TrxReader()
{
	if (trx_ != nullptr)
	{
		trx_->close();
		delete trx_;
		trx_ = nullptr;
	}
}

template <typename DT>
TrxReader<DT>::TrxReader(TrxReader &&other) noexcept : trx_(other.trx_)
{
	other.trx_ = nullptr;
}

template <typename DT>
TrxReader<DT> &TrxReader<DT>::operator=(TrxReader &&other) noexcept
{
	if (this != &other)
	{
		if (trx_ != nullptr)
		{
			trx_->close();
			delete trx_;
		}
		trx_ = other.trx_;
		other.trx_ = nullptr;
	}
	return *this;
}

template <typename Fn>
auto with_trx_reader(const std::string &path, Fn &&fn)
    -> decltype(fn(std::declval<TrxReader<float> &>(), TrxScalarType::Float32))
{
	const TrxScalarType dtype = detect_positions_scalar_type(path, TrxScalarType::Float32);
	switch (dtype)
	{
	case TrxScalarType::Float16:
	{
		TrxReader<Eigen::half> reader(path);
		return fn(reader, dtype);
	}
	case TrxScalarType::Float64:
	{
		TrxReader<double> reader(path);
		return fn(reader, dtype);
	}
	case TrxScalarType::Float32:
	default:
	{
		TrxReader<float> reader(path);
		return fn(reader, dtype);
	}
	}
}

template <typename DT>
void save(TrxFile<DT> &trx, const std::string filename, zip_uint32_t compression_standard)
{
	std::string ext = get_ext(filename);

	if (ext.size() > 0 && (strcmp(ext.c_str(), "zip") != 0 && strcmp(ext.c_str(), "trx") != 0))
	{
		throw std::invalid_argument("Unsupported extension." + ext);
	}

	TrxFile<DT> *copy_trx = trx.deepcopy();
	copy_trx->resize();
	std::string tmp_dir_name = copy_trx->_uncompressed_folder_handle;

	if (ext.size() > 0 && (strcmp(ext.c_str(), "zip") == 0 || strcmp(ext.c_str(), "trx") == 0))
	{
		int errorp;
		zip_t *zf;
		if ((zf = zip_open(filename.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp)) == NULL)
		{
			throw std::runtime_error("Could not open archive " + filename + ": " + strerror(errorp));
		}
		else
		{
			zip_from_folder(zf, tmp_dir_name, tmp_dir_name, compression_standard);
			if (zip_close(zf) != 0)
			{
				throw std::runtime_error("Unable to close archive " + filename + ": " + zip_strerror(zf));
			}
		}
	}
	else
	{
		struct stat sb;

		struct stat tmp_sb;
		if (stat(tmp_dir_name.c_str(), &tmp_sb) != 0 || !S_ISDIR(tmp_sb.st_mode))
		{
			throw std::runtime_error("Temporary TRX directory does not exist: " + tmp_dir_name);
		}

		if (stat(filename.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
		{
			if (rm_dir(filename.c_str()) != 0)
			{
				throw std::runtime_error("Could not remove existing directory " + filename);
			}
		}
		trx::fs::path dest_path(filename);
		if (dest_path.has_parent_path())
		{
			std::error_code ec;
			trx::fs::create_directories(dest_path.parent_path(), ec);
			if (ec)
			{
				throw std::runtime_error("Could not create output parent directory: " +
				                         dest_path.parent_path().string());
			}
		}
		copy_dir(tmp_dir_name.c_str(), filename.c_str());
		if (stat(filename.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
		{
			throw std::runtime_error("Failed to create output directory: " + filename);
		}
		const trx::fs::path header_path = dest_path / "header.json";
		if (!trx::fs::exists(header_path))
		{
			throw std::runtime_error("Missing header.json in output directory: " + header_path.string());
		}
		copy_trx->close();
	}
}

template <typename DT>
std::ostream &operator<<(std::ostream &out, const TrxFile<DT> &trx)
{

	out << "Header (header.json):\n";
	out << trx.header.dump();
	return out;
}
