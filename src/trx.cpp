#include <trx/trx.h>
#include <fstream>
#include <typeinfo>
#include <errno.h>
#include <algorithm>
#define SYSERROR() errno

//#define ZIP_DD_SIG 0x08074b50
//#define ZIP_CD_SIG 0x06054b50
using namespace Eigen;
using namespace std;

namespace trxmmap
{
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

	std::string get_extraction_dir()
	{
		std::string dir_template = "/tmp/trx_XXXXXX";
		char* extraction_dir;

		if (const char* trx_tmp_dir = std::getenv("TRX_TMPDIR")) {
			std::string trx_tmp_dir_s(trx_tmp_dir);
			if (trx_tmp_dir_s.compare("use_working_dir") == 0) {
				dir_template = "./trx_XXXXXX";
			} else {
				dir_template = trx_tmp_dir_s + "/trx_XXXXXX";
			}
		}
		extraction_dir = mkdtemp(&dir_template[0]);
		if (extraction_dir == NULL) {
			spdlog::error("Could not create temporary directory {}", dir_template);
			exit(1);
		}
		return std::string(extraction_dir);
	}

	int free_extraction_dir(std::string extraction_dir)
	{
		return rm_dir(extraction_dir.c_str());
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
	// TODO: check if there's a better way
	int _sizeof_dtype(std::string dtype)
	{
		// TODO: make dtypes enum??
		auto it = std::find(dtypes.begin(), dtypes.end(), dtype);
		int index = 0;
		if (it != dtypes.end())
		{
			index = std::distance(dtypes.begin(), it);
		}

		switch (index)
		{
		case 1:
			return 1;
		case 2:
			return sizeof(uint8_t);
		case 3:
			return sizeof(uint16_t);
		case 4:
			return sizeof(uint32_t);
		case 5:
			return sizeof(uint64_t);
		case 6:
			return sizeof(int8_t);
		case 7:
			return sizeof(int16_t);
		case 8:
			return sizeof(int32_t);
		case 9:
			return sizeof(int64_t);
		case 10:
			return sizeof(float);
		case 11:
			return sizeof(double);
		default:
			return sizeof(half); // setting this as default for now but a better solution is needed
		}
	}

	std::string _get_dtype(std::string dtype)
	{
		char dt = dtype.back();
		switch (dt)
		{
		case 'b':
			return "bit";
		case 'h':
			return "uint8";
		case 't':
			return "uint16";
		case 'j':
			return "uint32";
		case 'm':
			return "uint64";
		case 'a':
			return "int8";
		case 's':
			return "int16";
		case 'i':
			return "int32";
		case 'l':
			return "int64";
		case 'f':
			return "float32";
		case 'd':
			return "float64";
		default:
			return "float16"; // setting this as default for now but a better solution is needed
		}
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

	bool _is_dtype_valid(std::string &ext)
	{
		if (ext.compare("bit") == 0)
			return true;
		if (std::find(trxmmap::dtypes.begin(), trxmmap::dtypes.end(), ext) != trxmmap::dtypes.end())
			return true;
		return false;
	}

	json load_header(zip_t *zfolder)
	{
		// load file
		zip_file_t *zh = zip_fopen(zfolder, "header.json", ZIP_FL_UNCHANGED);

		// read data from file in chunks of 255 characters until data is fully loaded
		int buff_len = 255 * sizeof(char);
		char *buffer = (char *)malloc(buff_len);

		std::string jstream = "";
		zip_int64_t nbytes;
		while ((nbytes = zip_fread(zh, buffer, buff_len - 1)) > 0)
		{
			if (buffer != NULL)
			{
				jstream += string(buffer, nbytes);
			}
		}

		free(zh);
		free(buffer);

		// convert jstream data into Json.
		auto root = json::parse(jstream);
		return root;
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

	mio::shared_mmap_sink _create_memmap(std::string &filename, std::tuple<int, int> &shape, std::string mode, std::string dtype, long long offset)
	{
		if (dtype.compare("bool") == 0)
		{
			std::string ext = "bit";
			filename.replace(filename.size() - 4, 3, ext);
			filename.pop_back();
		}

		long filesize = std::get<0>(shape) * std::get<1>(shape) * _sizeof_dtype(dtype);
		// if file does not exist, create and allocate it

		struct stat buffer;
		if (stat(filename.c_str(), &buffer) != 0)
		{
			allocate_file(filename, filesize);
		}

		// std::error_code error;

		mio::shared_mmap_sink rw_mmap(filename, offset, filesize);

		return rw_mmap;
	}

	// TODO: support FORTRAN ORDERING
	// template <typename Derived>

	json assignHeader(json root)
	{
		json header = root;
		// MatrixXf affine(4, 4);
		// RowVectorXi dimensions(3);

		// for (int i = 0; i < 4; i++)
		// {
		// 	for (int j = 0; j < 4; j++)
		// 	{
		// 		affine << root["VOXEL_TO_RASMM"][i][j].asFloat();
		// 	}
		// }

		// for (int i = 0; i < 3; i++)
		// {
		// 	dimensions[i] << root["DIMENSIONS"][i].asUInt();
		// }
		// header["VOXEL_TO_RASMM"] = affine;
		// header["DIMENSIONS"] = dimensions;
		// header["NB_VERTICES"] = (int)root["NB_VERTICES"].asUInt();
		// header["NB_STREAMLINES"] = (int)root["NB_STREAMLINES"].asUInt();

		return header;
	}

	void get_reference_info(std::string reference, const MatrixXf &affine, const RowVectorXi &dimensions)
	{
		// TODO: find a library to use for nifti and trk (MRtrix??)
		//  if (reference.find(".nii") != std::string::npos)
		//  {
		//  }
		if (reference.find(".trk") != std::string::npos)
		{
			// TODO: Create exception class
			std::cout << "Trk reference not implemented" << std::endl;
			std::exit(1);
		}
		else
		{
			// TODO: Create exception class
			std::cout << "Trk reference not implemented" << std::endl;
			std::exit(1);
		}
	}

	void copy_dir(const char *src, const char *dst)
	{
		DIR *dir;
		struct dirent *entry;

		if (!(dir = opendir(src)))
			return;

		if (mkdir(dst, S_IRWXU) != 0)
		{
			spdlog::error("Could not create directory {}", dst);
		}

		while ((entry = readdir(dir)) != NULL)
		{
			if (entry->d_type == DT_DIR)
			{
				char path[1024];
				if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
					continue;
				char dstpath[1024];
				snprintf(path, sizeof(path), "%s%s%s", src, SEPARATOR.c_str(), entry->d_name);
				snprintf(dstpath, sizeof(dstpath), "%s%s%s", dst, SEPARATOR.c_str(), entry->d_name);
				copy_dir(path, dstpath);
			}
			else
			{
				char srcfile[1024];
				char dstfile[1024];
				snprintf(srcfile, sizeof(srcfile), "%s%s%s", src, SEPARATOR.c_str(), entry->d_name);
				snprintf(dstfile, sizeof(dstfile), "%s%s%s", dst, SEPARATOR.c_str(), entry->d_name);
				copy_file(srcfile, dstfile);
			}
		}
		closedir(dir);
	}

	// modified from:https://stackoverflow.com/a/7267734
	void copy_file(const char *src, const char *dst)
	{
		int src_fd, dst_fd, n, err;
		unsigned char buffer[4096];

		src_fd = open(src, O_RDONLY);
		dst_fd = open(dst, O_CREAT | O_WRONLY, 0666); // maybe keep original permissions?

		while (1)
		{
			err = read(src_fd, buffer, 4096);
			if (err == -1)
			{
				printf("Error reading file.\n");
				exit(1);
			}
			n = err;

			if (n == 0)
				break;

			err = write(dst_fd, buffer, n);
			if (err == -1)
			{
				printf("Error writing to file.\n");
				exit(1);
			}
		}
		close(src_fd);
		close(dst_fd);
	}
	int rm_dir(const char *d)
	{

		DIR *dir;
		struct dirent *entry;

		if (!(dir = opendir(d)))
			return -1;

		while ((entry = readdir(dir)) != NULL)
		{
			if (entry->d_type == DT_DIR)
			{
				char path[PATH_MAX];
				if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
					continue;
				snprintf(path, sizeof(path), "%s%s%s", d, SEPARATOR.c_str(), entry->d_name);
				rm_dir(path);
			}
			else
			{
				char fn[PATH_MAX];
				snprintf(fn, sizeof(fn), "%s%s%s", d, SEPARATOR.c_str(), entry->d_name);
				if (remove(fn) != 0)
				{
					spdlog::error("Could not remove file {}", fn);
					return -1;
				}
			}
		}
		closedir(dir);
		return rmdir(d);
	}

	void zip_from_folder(zip_t *zf, const std::string root, const std::string directory, zip_uint32_t compression_standard)
	{
		DIR *dir;
		struct dirent *entry;

		if (!(dir = opendir(directory.c_str())))
			return;

		while ((entry = readdir(dir)) != NULL)
		{
			if (entry->d_type == DT_DIR)
			{
				char path[PATH_MAX];
				if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
					continue;
				snprintf(path, sizeof(path), "%s%s%s", directory.c_str(), SEPARATOR.c_str(), entry->d_name);

				std::string zip_fname(path);
				zip_fname = rm_root(root, zip_fname);
				zip_dir_add(zf, zip_fname.c_str(), ZIP_FL_ENC_GUESS);
				zip_from_folder(zf, root, std::string(path), compression_standard);
			}
			else
			{
				std::string fn;
				char fullpath[PATH_MAX];

				snprintf(fullpath, sizeof(fullpath), "%s%s%s", directory.c_str(), SEPARATOR.c_str(), entry->d_name);
				fn = rm_root(root, std::string(fullpath));

				zip_source_t *s;

				if ((s = zip_source_file(zf, fullpath, 0, 0)) == NULL ||
				    zip_file_add(zf, fn.c_str(), s, ZIP_FL_ENC_UTF_8) < 0)
				{
					zip_source_free(s);
					spdlog::error("error adding file {}: {}", fn, zip_strerror(zf));
				}
			}
		}
		closedir(dir);
	}

	std::string rm_root(const std::string root, const std::string path)
	{

		std::size_t index;
		std::string stripped;

		index = path.find(root);
		if (index != std::string::npos)
		{
			stripped = path.substr(index + root.size() + 1, path.size() - index - root.size() - 1);
		}
		return stripped;
	}
};