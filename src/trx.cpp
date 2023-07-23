#include <trx/trx.h>
#include <fstream>
#include <typeinfo>
#include <errno.h>
#include <algorithm>
#define SYSERROR() errno

using namespace Eigen;
using namespace std;
namespace trx
{

	TrxFile::TrxFile(int nb_vertices, int nb_streamlines)
	{
		std::vector<std::vector<float>> affine(4);
		std::vector<uint16_t> dimensions(3);

		for (int i = 0; i < 4; i++)
		{
			affine[i] = {0, 0, 0, 0};
			affine[i][i] = 1;
		}
		dimensions = {1, 1, 1};
	}

	std::string get_ext(const std::string &str)
	{
		std::string ext = "";
		std::string delimeter = ".";

		if (str.rfind(delimeter) < str.length() - 1)
		{
			ext = str.substr(str.rfind(delimeter) + 1);
		}
		return ext;
	}

	bool _is_dtype_valid(std::string &ext)
	{
		return true;
		// if (ext.compare("bit") == 0)
		// 	return true;
		// if (std::find(trx::dtypes.begin(), trx::dtypes.end(), ext) != trx::dtypes.end())
		// 	return true;
		// return false;
	}

	std::string get_base(const std::string &delimiter, const std::string &str)
	{
		std::string token;

		if (str.rfind(delimiter) + 1 < str.length())
		{
			token = str.substr(str.rfind(delimiter) + 1);
		}
		else
		{
			token = str;
		}
		return token;
	}

	std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string filename)
	{

		// TODO: won't work on windows and not validating OS type
		std::string base = get_base("/", filename);

		size_t num_splits = std::count(base.begin(), base.end(), '.');
		int dim;

		if (num_splits != 1 and num_splits != 2)
		{
			throw std::invalid_argument("Invalid filename");
		}

		std::string ext = get_ext(filename);

		base = base.substr(0, base.length() - ext.length() - 1);

		if (num_splits == 1)
		{
			dim = 1;
		}
		else
		{
			int pos = base.find_last_of(".");
			dim = std::stoi(base.substr(pos + 1, base.size()));
			base = base.substr(0, pos);
		}

		bool is_valid = _is_dtype_valid(ext);

		if (is_valid == false)
		{
			// TODO: make formatted string and include provided extension name
			throw std::invalid_argument("Unsupported file extension");
		}

		std::tuple<std::string, int, std::string> output{base, dim, ext};

		return output;
	}

	Dtype _get_dtype(std::string dtype)
	{
		for (int i = 0; i < sizeof(AllDtypes); i++)
		{
			if (dtype == AllDtypes[i].name)
			{
				return (Dtype)i;
			}
		}

		return (Dtype)-1;
	}

	int _sizeof_dtype(std::string dtype)
	{
		for (int i = 0; i < sizeof(AllDtypes); i++)
		{
			if (dtype == AllDtypes[i].name)
			{
				return AllDtypes[i].size;
			}
		}

		return -1;
	}

	void populate_fps(const char *name, std::map<std::string, std::tuple<long long, long long>> &files_pointer_size)
	{
		DIR *dir;
		struct dirent *entry;

		if (!(dir = opendir(name)))
			return;

		while ((entry = readdir(dir)) != NULL)
		{
			if (entry->d_type == DT_DIR)
			{
				char path[1024];
				if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
					continue;
				snprintf(path, sizeof(path), "%s/%s", name, entry->d_name);
				populate_fps(path, files_pointer_size);
			}
			else
			{
				std::string filename(entry->d_name);
				std::string root(name);
				std::string elem_filename = root + SEPARATOR + filename;
				std::string ext = get_ext(elem_filename);

				if (strcmp(ext.c_str(), "json") == 0)
				{
					continue;
				}

				if (!_is_dtype_valid(ext))
				{
					throw std::invalid_argument(std::string("The dtype of ") + elem_filename + std::string(" is not supported"));
				}

				if (strcmp(ext.c_str(), "bit") == 0)
				{
					ext = "bool";
				}

				int dtype_size = _sizeof_dtype(ext);

				struct stat sb;
				unsigned long size = 0;

				if (stat(elem_filename.c_str(), &sb) == 0)
				{
					size = sb.st_size / dtype_size;
				}

				if (sb.st_size % dtype_size == 0)
				{
					files_pointer_size[elem_filename] = std::make_tuple(0, size);
				}
				else if (sb.st_size == 1)
				{
					files_pointer_size[elem_filename] = std::make_tuple(0, 0);
				}
				else
				{
					std::invalid_argument("Wrong size of datatype");
				}
			}
		}
		closedir(dir);
	}

	TrxFile *load_from_directory(std::string path)
	{
		std::string directory = (std::string)canonicalize_file_name(path.c_str());
		std::string header_name = directory + SEPARATOR + "header.json";

		std::ifstream header_file(header_name);
		json header;
		header_file >> header;
		header_file.close();

		std::map<std::string, std::tuple<long long, long long>> files_pointer_size;
		populate_fps(directory.c_str(), files_pointer_size);
		
		return TrxFile::_create_trx_from_pointer(header, files_pointer_size, "", directory);
	}

	std::string * _get_pointer_by_name(
		std::string lookup,
		std::map<std::string, std::tuple<long long, long long>> dict_pointer_size
	)
	{
		for(std::map<std::string, std::tuple<long long, long long>>::iterator iter = dict_pointer_size.begin(); iter != dict_pointer_size.end(); ++iter)
		{
			std::string k =  iter->first;
			if (strncmp(k.c_str(), lookup.c_str(), lookup.size())) {
				return new std::string(iter->first.c_str());
			}
		}
		return NULL;
	}
	
	mio::shared_mmap_sink _create_memmap(std::string &filename, std::vector<int> shape, std::string mode, std::string dtype, long long offset)
	{

		if (dtype.compare("bool") == 0)
		{
			std::string ext = "bit";
			filename.replace(filename.size() - 4, 3, ext);
			filename.pop_back();
		}

		long filesize = _sizeof_dtype(dtype);
		for(std::vector<int>::iterator it = shape.begin(); it != shape.end(); ++it)
			filesize += *it;

		struct stat buffer;
		if (stat(filename.c_str(), &buffer) != 0)
		{
			allocate_file(filename, filesize);
		}

		mio::shared_mmap_sink rw_mmap(filename, offset, filesize);
		return rw_mmap;
	}

	uint32_t* _compute_lengths(uint64_t* offsets, int nb_streamlines, int nb_vertices)
	{
		if (nb_streamlines > 1)
		{
			int last_elem_pos = _dichotomic_search(offsets);
			Matrix<uint32_t, Dynamic, 1> lengths;

			if (last_elem_pos == nb_streamlines - 1)
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
		if (nb_streamlines == 1)
		{
			Matrix<uint32_t, 1, 1, RowMajor> lengths(nb_vertices);
			return lengths;
		}

		Matrix<uint32_t, 1, 1, RowMajor> lengths(0);
		return lengths;
	}


	void allocate_file(const std::string &path, const int size)
	{
		std::ofstream file(path);
		if (file.is_open())
		{
			std::string s(size, float(0));
			file << s;
			file.flush();
			file.close();
		}
		else
		{
			std::cerr << "Failed to allocate file : " << SYSERROR() << std::endl;
		}
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

	TrxFile *TrxFile::_create_trx_from_pointer(
		json header,
		std::map<std::string, std::tuple<long long, long long>> dict_pointer_size,
		std::string root_zip, std::string root
	)
	{
		TrxFile *trx = new TrxFile();
		trx->header = header;

		std::string filename;

		std::string* positions = _get_pointer_by_name(
			"positions.",
			dict_pointer_size
		);
		if (positions == NULL) {
			throw std::invalid_argument("positions not found.");
		}
		
		std::string* offsets = _get_pointer_by_name(
			"offsets.",
			dict_pointer_size
		);
		if (offsets == NULL) {
			throw std::invalid_argument("offsets not found.");
		}

		int nb_vertices(int(trx->header["NB_VERTICES"]));
		int nb_streamlines(int(trx->header["NB_STREAMLINES"]));

		auto positions_filename = *positions;
		auto positions_dtype = std::get<2>(_split_ext_with_dimensionality(positions_filename));
		std::tuple<long long, long long> positions_offset_size = dict_pointer_size[positions_filename];

		auto offsets_filename = *offsets;
		auto offsets_dtype = std::get<2>(_split_ext_with_dimensionality(offsets_filename));
		std::tuple<long long, long long> offsets_offset_size = dict_pointer_size[offsets_filename];

		trx->streamlines = new ArraySequence(_get_dtype(positions_dtype));
		trx->streamlines->mmap_positions = _create_memmap(positions_filename, std::vector<int>{nb_vertices, 3}, "r+", positions_dtype, std::get<0>(positions_offset_size));
		trx->streamlines->mmap_offsets = _create_memmap(offsets_filename, std::vector<int>{nb_vertices}, "r+", offsets_dtype, std::get<0>(offsets_offset_size));

		uint64_t* offsets = trx->streamlines->_offsets;
		trx->streamlines->_lengths = _compute_lengths(
			offsets, nb_streamlines, nb_vertices
		);

		for (auto entry = dict_pointer_size.begin(); entry != dict_pointer_size.end(); entry++)
		{
			std::string elem_filename = entry->first;

			if (root_zip.size() > 0)
			{
				filename = root_zip;
			}
			else
			{
				filename = elem_filename;
			}

			std::string folder = std::string(dirname(const_cast<char *>(strdup(elem_filename.c_str()))));

			std::tuple<std::string, int, std::string> base_tuple = _split_ext_with_dimensionality(elem_filename);
			std::string base(std::get<0>(base_tuple));
			int dim = std::get<1>(base_tuple);
			std::string ext(std::get<2>(base_tuple));

			Dtype dtype = _get_dtype(ext);

			long long mem_adress = std::get<0>(x->second);
			long long size = std::get<1>(x->second);

			std::string stripped = root;
			if (stripped.rfind("/") == stripped.size() - 1)
			{
				stripped = stripped.substr(0, stripped.size() - 1);
			}

			if (root.compare("") != 0 && folder.rfind(stripped, stripped.size()) == 0)
			{
				folder = folder.replace(0, root.size(), "");
				if (folder[0] == SEPARATOR.c_str()[0])
				{
					folder = folder.substr(1, folder.size());
				}
			}

			if (folder.compare("dps") == 0)
			{
				Dtype dtype = _get_dtype(ext);

				std::tuple<int, int> shape;
				trx->data_per_streamline[base] = new MMappedMatrix(dtype);
				int nb_scalar = size / nb_streamlines;

				if (size % nb_streamlines != 0 || nb_scalar != dim)
				{

					throw std::invalid_argument("Wrong dps size/dimensionality");
				}
				else
				{
					shape = std::make_tuple(nb_streamlines, nb_scalar);
				}
				trx->data_per_streamline[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);
			}

			else if (folder.compare("dpv") == 0)
			{
				std::tuple<int, int> shape;
				int nb_scalar = size / nb_vertices;

				if (size % nb_vertices != 0 || nb_scalar != dim)
				{

					throw std::invalid_argument("Wrong dpv size/dimensionality");
				}
				else
				{
					shape = std::make_tuple(trx->header["NB_VERTICES"], nb_scalar);
				}

				Dtype dtype = _get_dtype(ext);
				trx->data_per_vertex[base] = new ArraySequence(dtype);
				trx->data_per_vertex[base]->mmap_pos = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);
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

				Dtype dtype = _get_dtype(ext);

				trx->data_per_group[sub_folder][data_name] = new MMappedMatrix(dtype);
				trx->data_per_group[sub_folder][data_name]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);
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
				Dtype dtype = _get_dtype(ext);
				trx->groups[base] = new MMappedMatrix(dtype);
				trx->groups[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext, mem_adress);
			}
			else
			{
				spdlog::error("{} is not part of a valid structure.", elem_filename);
			}
		}

		return trx;
	}

}