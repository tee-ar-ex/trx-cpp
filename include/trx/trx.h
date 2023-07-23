#ifndef TRX_H // include guard
#define TRX_H

#include <iostream>
#include <fstream>
#include <zip.h>
#include <string.h>
#include <dirent.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <math.h>
#include <libgen.h>
#include <Eigen/Core>
#include <filesystem>

#include <mio/mmap.hpp>
#include <mio/shared_mmap.hpp>

#ifndef SPDLOG_FMT_EXTERNAL
#define SPDLOG_FMT_EXTERNAL
#endif
#include "spdlog/spdlog.h"

using namespace Eigen;
using json = nlohmann::json;

namespace trx
{

	const std::string SEPARATOR = "/";

	struct DtypeInfo {
		std::string name;
		std::size_t size;
	};

	enum Dtype
	{
		bit,
		uint8,
		uint16,
		uint32,
		uint64,
		int8,
		int16,
		int32,
		int64,
		float16,
		float32,
		float64
	};

	static const DtypeInfo AllDtypes[] = {
		{"float16", sizeof(half)},
		{"float32", sizeof(float)},
		{"float64", sizeof(double)},
		{"bit", sizeof(bool)},
		{"uint8", sizeof(uint8_t)},
		{"uint16", sizeof(uint16_t)},
		{"uint32", sizeof(uint32_t)},
		{"uint64", sizeof(uint64_t)},
		{"int8", sizeof(int8_t)},
		{"int16", sizeof(int16_t)},
		{"int32", sizeof(int32_t)},
		{"int64", sizeof(int64_t)},
	};

	struct ArraySequence
	{
		Dtype _dtype;
		void *_data;
		uint64_t *_offsets;
		uint32_t *_lengths;
		mio::shared_mmap_sink mmap_positions;
		mio::shared_mmap_sink mmap_offsets;

		ArraySequence(Dtype dtype) : _dtype(dtype), _data(NULL), _offsets(NULL), _lengths(NULL){};
	};

	struct MMappedMatrix
	{
		Dtype _dtype;
		void *_data;
		std::tuple<int, int> _shape;
		mio::shared_mmap_sink *_mmap;

		MMappedMatrix(Dtype dtype) : _dtype(dtype), _data(NULL){};
	};


	class TrxFile
	{
	public:
		json header;

		ArraySequence * streamlines;

		TrxFile(int nb_vertices = 0, int nb_streamlines = 0);
		static TrxFile *_create_trx_from_pointer(json header, std::map<std::string, std::tuple<long long, long long>> dict_pointer_size, std::string root_zip = "", std::string root = "");
	};

	std::string get_ext(const std::string &str);
	bool _is_dtype_valid(std::string &ext);
	std::string get_base(const std::string &delimiter, const std::string &str);
	std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string filename);
	Dtype _get_dtype(std::string dtype);
	int _sizeof_dtype(std::string dtype);
	void populate_fps(const char *name, std::map<std::string, std::tuple<long long, long long>> &files_pointer_size);

	void allocate_file(const std::string &path, const int size);
	TrxFile *load_from_directory(std::string path);
	std::string * _get_pointer_by_name(
		std::string lookup,
		std::map<std::string, std::tuple<long long, long long>> dict_pointer_size
	);
}
#endif