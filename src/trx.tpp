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
std::string _generate_filename_from_data(const MatrixBase<DT> &arr, std::string filename)
{

	std::string base, ext;

	base = filename; // get_base(SEPARATOR, filename);
	ext = get_ext(filename);

	if (ext.size() != 0)
	{
		spdlog::warn("Will overwrite provided extension if needed.");
		base = base.substr(0, base.length() - ext.length() - 1);
	}

	// A very backwards way of getting the datatype
	std::string eigen_dt = typeid(arr.array().matrix().data()).name();
	std::string dt = _get_dtype(eigen_dt);

	int n_rows = arr.rows();
	int n_cols = arr.cols();

	std::string new_filename;
	if (n_cols == 1)
	{
		int buffsize = filename.size() + dt.size() + 2;
		char buff[buffsize];
		snprintf(buff, sizeof(buff), "%s.%s", base.c_str(), dt.c_str());
		new_filename = buff;
	}
	else
	{
		int buffsize = filename.size() + dt.size() + n_cols + 3;
		char buff[buffsize];
		snprintf(buff, sizeof(buff), "%s.%i.%s", base.c_str(), n_cols, dt.c_str());
		new_filename = buff;
	}

	return new_filename;
}

template <typename DT>
Matrix<uint32_t, Dynamic, 1> _compute_lengths(const MatrixBase<DT> &offsets, int nb_vertices)
{
	if (offsets.size() > 1)
	{
		int last_elem_pos = _dichotomic_search(offsets);
		Matrix<uint32_t, Dynamic, 1> lengths;

		if (last_elem_pos == offsets.size() - 1)
		{
			Matrix<uint32_t, Dynamic, Dynamic> tmp(offsets.template cast<u_int32_t>());
			ediff1d(lengths, tmp, uint32_t(nb_vertices - offsets(last)));
		}
		else
		{
			Matrix<uint32_t, Dynamic, Dynamic> tmp(offsets.template cast<u_int32_t>());
			tmp(last_elem_pos + 1) = uint32_t(nb_vertices);
			ediff1d(lengths, tmp, 0);
			lengths(last_elem_pos + 1) = uint32_t(0);
		}
		return lengths;
	}
	if (offsets.size() == 1)
	{
		Matrix<uint32_t, 1, 1, RowMajor> lengths(nb_vertices);
		return lengths;
	}

	Matrix<uint32_t, 1, 1, RowMajor> lengths(0);
	return lengths;
}

template <typename DT>
int _dichotomic_search(const MatrixBase<DT> &x, int l_bound, int r_bound)
{
	if (l_bound == -1 && r_bound == -1)
	{
		l_bound = 0;
		r_bound = x.size() - 1;
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
				affine[i][j] = float(init_as->header["VOXEL_TO_RASMM"][i][j]);
			}
		}

		for (int i = 0; i < 3; i++)
		{
			dimensions[i] = uint16_t(init_as->header["DIMENSIONS"][i]);
		}
	}
	// TODO: add else if for get_reference_info
	else
	{
		spdlog::debug("No reference provided, using blank space attributes, please update them later.");

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

		spdlog::debug("Initializing empty TrxFile.");
		// will remove as completely unecessary. using as placeholders
		this->header = {};

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
		spdlog::debug("Preallocating TrxFile with size {} streamlines and {} vertices.", nb_streamlines, nb_vertices);
		TrxFile<DT> *trx = _initialize_empty_trx<DT>(nb_streamlines, nb_vertices, init_as);
		this->streamlines = trx->streamlines;
		this->groups = trx->groups;
		this->data_per_streamline = trx->data_per_streamline;
		this->data_per_vertex = trx->data_per_vertex;
		this->data_per_group = trx->data_per_group;
		this->_uncompressed_folder_handle = trx->_uncompressed_folder_handle;
		this->_copy_safe = trx->_copy_safe;
	}
	else
	{
		throw std::invalid_argument("You must declare both NB_VERTICES AND NB_STREAMLINES");
	}

	this->header["VOXEL_TO_RASMM"] = affine;
	this->header["DIMENSIONS"] = dimensions;
	this->header["NB_VERTICES"] = nb_vertices;
	this->header["NB_STREAMLINES"] = nb_streamlines;

	this->_copy_safe = true;
}

template <typename DT>
TrxFile<DT> *_initialize_empty_trx(int nb_streamlines, int nb_vertices, const TrxFile<DT> *init_as)
{
	TrxFile<DT> *trx = new TrxFile<DT>();

	char *dirname;
	char t[] = "/tmp/trx_XXXXXX";
	dirname = mkdtemp(t);

	std::string tmp_dir(dirname);

	spdlog::info("Temporary folder for memmaps: {}", tmp_dir);

	trx->header["NB_VERTICES"] = nb_vertices;
	trx->header["NB_STREAMLINES"] = nb_streamlines;

	std::string positions_dtype;
	std::string offsets_dtype;
	std::string lengths_dtype;

	if (init_as != NULL)
	{
		trx->header["VOXEL_TO_RASMM"] = init_as->header["VOXEL_TO_RASMM"];
		trx->header["DIMENSIONS"] = init_as->header["DIMENSIONS"];
		positions_dtype = _get_dtype(typeid(init_as->streamlines->_data).name());
		offsets_dtype = _get_dtype(typeid(init_as->streamlines->_offsets).name());
		lengths_dtype = _get_dtype(typeid(init_as->streamlines->_lengths).name());
	}
	else
	{
		positions_dtype = _get_dtype(typeid(half).name());
		offsets_dtype = _get_dtype(typeid(uint64_t).name());
		lengths_dtype = _get_dtype(typeid(uint32_t).name());
	}
	spdlog::debug("Initializing positions with dtype: {}", positions_dtype);
	spdlog::debug("Initializing offsets with dtype: {}", offsets_dtype);
	spdlog::debug("Initializing lengths with dtype: {}", lengths_dtype);

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

	std::tuple<int, int> shape_off = std::make_tuple(nb_streamlines, 1);

	trx->streamlines->mmap_off = trxmmap::_create_memmap(offsets_filename, shape_off, "w+", offsets_dtype);
	new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape_off), std::get<1>(shape_off));

	trx->streamlines->_lengths.resize(nb_streamlines);

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
			std::string dpv_dtype = _get_dtype(typeid(init_as->data_per_vertex.find(x.first)->second->_data).name());
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

			spdlog::debug("Initializing {} (dpv) with dtype: {}", x.first, dpv_dtype);

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
			std::string dps_dtype = _get_dtype(typeid(init_as->data_per_streamline.find(x.first)->second->_matrix).name());
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

			spdlog::debug("Initializing {} (dps) with dtype: {}", x.first, dps_dtype);

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

	trx->_uncompressed_folder_handle = tmp_dir;

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

		std::string folder = std::string(dirname(const_cast<char *>(strdup(elem_filename.c_str()))));

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
			if (size != int(trx->header["NB_VERTICES"]) * 3 || dim != 3)
			{

				throw std::invalid_argument("Wrong data size/dimensionality");
			}

			std::tuple<int, int> shape = std::make_tuple(trx->header["NB_VERTICES"], 3);
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
			if (size != int(trx->header["NB_STREAMLINES"]) || dim != 1)
			{

				throw std::invalid_argument("Wrong offsets size/dimensionality");
			}

			std::tuple<int, int> shape = std::make_tuple(trx->header["NB_STREAMLINES"], 1);
			trx->streamlines->mmap_off = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);

			new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape), std::get<1>(shape));

			// TODO : adapt compute_lengths to accept a map
			Matrix<uint64_t, Dynamic, 1> offsets;
			offsets = trx->streamlines->_offsets;
			trx->streamlines->_lengths = _compute_lengths(offsets, int(trx->header["NB_VERTICES"]));
		}

		else if (folder.compare("dps") == 0)
		{
			std::tuple<int, int> shape;
			trx->data_per_streamline[base] = new MMappedMatrix<DT>();
			int nb_scalar = size / int(trx->header["NB_STREAMLINES"]);

			if (size % int(trx->header["NB_STREAMLINES"]) != 0 || nb_scalar != dim)
			{

				throw std::invalid_argument("Wrong dps size/dimensionality");
			}
			else
			{
				shape = std::make_tuple(trx->header["NB_STREAMLINES"], nb_scalar);
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
			int nb_scalar = size / int(trx->header["NB_VERTICES"]);

			if (size % int(trx->header["NB_VERTICES"]) != 0 || nb_scalar != dim)
			{

				throw std::invalid_argument("Wrong dpv size/dimensionality");
			}
			else
			{
				shape = std::make_tuple(trx->header["NB_VERTICES"], nb_scalar);
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

			new (&(trx->data_per_vertex[base]->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape), std::get<1>(shape));
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

			std::string data_name = std::string(basename(const_cast<char *>(base.c_str())));
			std::string sub_folder = std::string(basename(const_cast<char *>(folder.c_str())));

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
			spdlog::error("{} is not part of a valid structure.", elem_filename);
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
	char *dirname;
	char t[] = "/tmp/trx_XXXXXX";
	dirname = mkdtemp(t);

	std::string tmp_dir(dirname);

	std::string header = tmp_dir + SEPARATOR + "header.json";
	std::ofstream out_json(header);

	// TODO: Definitely a better way to deepcopy
	json tmp_header = json::parse(this->header.dump());

	ArraySequence<DT> *to_dump = new ArraySequence<DT>();
	// TODO: Verify that this is indeed a deep copy
	new (&(to_dump->_data)) Matrix<DT, Dynamic, Dynamic, RowMajor>(this->streamlines->_data);
	new (&(to_dump->_offsets)) Matrix<uint64_t, Dynamic, Dynamic, RowMajor>(this->streamlines->_offsets);
	new (&(to_dump->_lengths)) Matrix<uint32_t, Dynamic, 1>(this->streamlines->_lengths);

	if (!this->_copy_safe)
	{
		tmp_header["NB_STREAMLINES"] = to_dump->_offsets.size();
		tmp_header["NB_VERTICES"] = to_dump->_data.size() / 3;
	}
	if (out_json.is_open())
	{
		out_json << std::setw(4) << tmp_header << std::endl;
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
			spdlog::error("Could not create directory {}", dpv_dirname);
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
			spdlog::error("Could not create directory {}", dps_dirname);
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
			spdlog::error("Could not create directory {}", groups_dirname);
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
						spdlog::error("Could not create directory {}", dpg_dirname);
					}
				}
				if (stat(dpg_subdirname.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
				{
					if (mkdir(dpg_subdirname.c_str(), S_IRWXU) != 0)
					{
						spdlog::error("Could not create directory {}", dpg_subdirname);
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

	this->streamlines->_data(seq(pts_start, pts_end - 1), all) = trx->streamlines->_data(seq(0, curr_pts_len - 1), all);
	this->streamlines->_offsets(seq(strs_start, strs_end - 1), all) = (trx->streamlines->_offsets(seq(0, curr_strs_len - 1), all).array() + pts_start).matrix();
	this->streamlines->_lengths(seq(strs_start, strs_end - 1), all) = trx->streamlines->_lengths(seq(0, curr_strs_len - 1), all);

	for (auto const &x : this->data_per_vertex)
	{
		this->data_per_vertex[x.first]->_data(seq(pts_start, pts_end - 1), all) = trx->data_per_vertex[x.first]->_data(seq(0, curr_pts_len - 1), all);
		new (&(this->data_per_vertex[x.first]->_offsets)) Map<Matrix<uint64_t, Dynamic, Dynamic>>(trx->data_per_vertex[x.first]->_offsets.data(), trx->data_per_vertex[x.first]->_offsets.rows(), trx->data_per_vertex[x.first]->_offsets.cols());
		this->data_per_vertex[x.first]->_lengths = trx->data_per_vertex[x.first]->_lengths;
	}

	for (auto const &x : this->data_per_streamline)
	{
		this->data_per_streamline[x.first]->_matrix(seq(strs_start, strs_end - 1), all) = trx->data_per_streamline[x.first]->_matrix(seq(0, curr_strs_len - 1), all);
	}

	return std::make_tuple(strs_end, pts_end);
}

template <typename DT>
void TrxFile<DT>::close()
{
	if (this->_uncompressed_folder_handle != "")
	{
		this->_uncompressed_folder_handle = "";
	}

	*this = TrxFile<DT>(); // probably dangerous to do
	spdlog::debug("Deleted memmaps and initialized empty TrxFile.");
}

template <typename DT>
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
		spdlog::info("Resizing (down) memmaps, less streamlines than it actually contains");
	}

	if (nb_vertices == -1)
	{
		ptrs_end = this->streamlines->_lengths(all, 0).sum();
		nb_vertices = ptrs_end;
	}
	else if (nb_vertices < ptrs_end)
	{
		spdlog::warn("Cannot resize (down) vertices for consistency.");
		return;
	}

	if (nb_streamlines == -1)
	{
		nb_streamlines = strs_end;
	}

	if (nb_streamlines == this->header["NB_STREAMLINES"] && nb_vertices == this->header["NB_VERTICES"])
	{
		spdlog::debug("TrxFile of the right size, no resizing.");
		return;
	}

	TrxFile<DT> *trx = _initialize_empty_trx(nb_streamlines, nb_vertices, this);

	spdlog::info("Resizing streamlines from size {} to {}", this->streamlines->_lengths.size(), nb_streamlines);
	spdlog::info("Resizing vertices from size {} to  {}", this->streamlines->_data(all, 0).size(), nb_vertices);

	if (nb_streamlines < this->header["NB_STREAMLINES"])
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
			spdlog::error("Could not create directory {}", group_dir);
		}

		for (auto const &x : this->groups)
		{
			std::string group_dtype = _get_dtype(typeid(x.second->_matrix).name());
			std::string group_name = group_dir + x.first + "." + group_dtype;

			int ori_length = this->groups[x.first]->_matrix.size();

			std::vector<int> keep_rows;
			std::vector<int> keep_cols = {0};

			// Slicing
			for (int i = 0; i < x.second->_matrix.rows(); ++i)
			{
				for (int j = 0; j < x.second->_matrix.cols(); ++j)
				{
					if (x.second->_matrix(i, j) < strs_end)
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

			spdlog::debug("{} group went from {} items to {}", x.first, ori_length, tmp.size());

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
			spdlog::error("Could not create directory {}", dpg_dir);
		}

		for (auto const &x : this->data_per_group)
		{
			std::string dpg_subdir = dpg_dir + x.first;
			struct stat sb;

			if (stat(dpg_subdir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
			{
				if (mkdir(dpg_subdir.c_str(), S_IRWXU) != 0)
				{
					spdlog::error("Could not create directory {}", dpg_subdir);
				}
			}

			if (trx->data_per_group.find(x.first) == trx->data_per_group.end())
			{
				trx->data_per_group[x.first] = {};
			}

			for (auto const &y : this->data_per_group[x.first])
			{
				std::string dpg_dtype = _get_dtype(typeid(this->data_per_group[x.first][y.first]->_matrix).name());
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
	// TODO: check error values
	int *errorp;
	zip_t *zf = zip_open(filename.c_str(), 0, errorp);
	json header = load_header(zf);

	std::map<std::string, std::tuple<long long, long long>> file_pointer_size;
	long long global_pos = 0;
	long long mem_address = 0;

	int num_entries = zip_get_num_entries(zf, ZIP_FL_UNCHANGED);

	for (int i = 0; i < num_entries; ++i)
	{
		std::string elem_filename = zip_get_name(zf, i, ZIP_FL_UNCHANGED);

		zip_stat_t sb;
		zip_file_t *zft;

		if (zip_stat(zf, elem_filename.c_str(), ZIP_FL_UNCHANGED, &sb) != 0)
		{
			return NULL;
		}

		global_pos += 30 + elem_filename.size();

		size_t lastdot = elem_filename.find_last_of(".");

		if (lastdot == std::string::npos)
		{
			global_pos += sb.comp_size;
			continue;
		}
		std::string ext = elem_filename.substr(lastdot + 1, std::string::npos);

		// apparently all zip directory names end with a slash. may be a better way
		if (ext.compare("json") == 0 || elem_filename.rfind("/") == elem_filename.size() - 1)
		{
			global_pos += sb.comp_size;
			continue;
		}

		if (!_is_dtype_valid(ext))
		{
			global_pos += sb.comp_size;
			continue;
			// maybe throw error here instead?
			// throw std::invalid_argument("The dtype is not supported");
		}

		if (ext.compare("bit") == 0)
		{
			ext = "bool";
		}

		// get file stats

		// std::ifstream file(filename, std::ios::binary);
		// file.seekg(global_pos);

		// unsigned char signature[4] = {0};
		// const unsigned char local_sig[4] = {0x50, 0x4b, 0x03, 0x04};
		// file.read((char *)signature, sizeof(signature));

		// if (memcmp(signature, local_sig, sizeof(signature)) == 0)
		// {
		// 	global_pos += 30;
		// 	// global_pos += sb.comp_size + elem_filename.size();
		// }

		long long size = sb.size / _sizeof_dtype(ext);
		mem_address = global_pos;
		file_pointer_size[elem_filename] = {mem_address, size};
		global_pos += sb.comp_size;
	}
	return TrxFile<DT>::_create_trx_from_pointer(header, file_pointer_size, filename);
}

template <typename DT>
TrxFile<DT> *load_from_directory(std::string path)
{
	std::string directory = (std::string)canonicalize_file_name(path.c_str());
	std::string header_name = directory + SEPARATOR + "header.json";

	// TODO: add check to verify that it's open
	std::ifstream header_file(header_name);
	json header;
	header_file >> header;
	header_file.close();

	std::map<std::string, std::tuple<long long, long long>> files_pointer_size;
	populate_fps(directory.c_str(), files_pointer_size);

	return TrxFile<DT>::_create_trx_from_pointer(header, files_pointer_size, "", directory);
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
			spdlog::error("Could not open file {} due to error: {}", filename, strerror(errorp));
		}
		else
		{
			zip_from_folder(zf, tmp_dir_name, tmp_dir_name, compression_standard);
			if (zip_close(zf) != 0)
			{
				spdlog::error("Unable to close archive {} due to : {}", filename, strerror(errorp));
			}
		}
	}
	else
	{
		struct stat sb;

		if (stat(filename.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
		{
			if (rm_dir(filename.c_str()) != 0)
			{
				spdlog::error("Could not remove existing directory {}", filename);
			}
		}
		copy_dir(tmp_dir_name.c_str(), filename.c_str());
		copy_trx->close();
	}
}

template <typename DT>
std::ostream &operator<<(std::ostream &out, const TrxFile<DT> &trx)
{

	out << "Header (header.json):\n";
	out << trx.header.dump(4);
	return out;
}
