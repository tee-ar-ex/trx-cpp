#include <gtest/gtest.h>
#include <trx/trx.h>
#include <trx/filesystem.h>
#include <zip.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace Eigen;
using namespace trxmmap;
namespace fs = trx::fs;

namespace
{
	std::string get_test_data_root()
	{
		const char *env = std::getenv("TRX_TEST_DATA_DIR");
		if (env == nullptr || std::string(env).empty())
		{
			return {};
		}
		return std::string(env);
	}

	fs::path resolve_gold_standard_dir(const std::string &root_dir)
	{
		fs::path root(root_dir);
		fs::path gs_dir = root / "gold_standard";
		if (fs::exists(gs_dir))
		{
			return gs_dir;
		}
		return root;
	}

	fs::path require_gold_standard_dir()
	{
		const auto root = get_test_data_root();
		if (root.empty())
		{
			throw std::runtime_error("TRX_TEST_DATA_DIR not set");
		}

		const auto gs_dir = resolve_gold_standard_dir(root);
		const fs::path gs_trx = gs_dir / "gs.trx";
		const fs::path gs_dir_trx = gs_dir / "gs_fldr.trx";
		const fs::path gs_coords = gs_dir / "gs_rasmm_space.txt";
		if (!fs::exists(gs_trx))
		{
			throw std::runtime_error("Missing gold_standard gs.trx");
		}
		if (!fs::exists(gs_dir_trx))
		{
			throw std::runtime_error("Missing gold_standard gs_fldr.trx");
		}
		if (!fs::exists(gs_coords))
		{
			throw std::runtime_error("Missing gold_standard gs_rasmm_space.txt");
		}
		return gs_dir;
	}

	fs::path make_temp_test_dir(const std::string &prefix)
	{
		std::error_code ec;
		auto base = fs::temp_directory_path(ec);
		if (ec)
		{
			throw std::runtime_error("Failed to get temp directory: " + ec.message());
		}

		static std::mt19937_64 rng(std::random_device{}());
		std::uniform_int_distribution<uint64_t> dist;

		for (int attempt = 0; attempt < 100; ++attempt)
		{
			fs::path candidate = base / (prefix + "_" + std::to_string(dist(rng)));
			std::error_code dir_ec;
			if (fs::create_directory(candidate, dir_ec))
			{
				return candidate;
			}
			if (dir_ec && dir_ec != std::errc::file_exists)
			{
				throw std::runtime_error("Failed to create temporary directory: " + dir_ec.message());
			}
		}
		throw std::runtime_error("Unable to create unique temporary directory");
	}

	std::string normalize_path(const std::string &path)
	{
		std::string out = path;
		std::replace(out.begin(), out.end(), '\\', '/');
		return out;
	}

	std::string normalize_path(const fs::path &path)
	{
		return normalize_path(path.string());
	}

	bool is_directory(const fs::path &path)
	{
		std::error_code ec;
		return fs::is_directory(path, ec) && !ec;
	}

	bool is_regular_file(const fs::path &path)
	{
		struct stat sb;
		if (stat(path.c_str(), &sb) != 0)
		{
			return false;
		}
		return S_ISREG(sb.st_mode);
	}

	Matrix<float, Dynamic, Dynamic, RowMajor> load_rasmm_coords(const fs::path &path)
	{
		std::ifstream in(path.string());
		if (!in.is_open())
		{
			throw std::runtime_error("Failed to open coordinate file: " + path.string());
		}

		std::vector<float> values;
		float v;
		while (in >> v)
		{
			values.push_back(v);
		}
		if (values.size() % 3 != 0)
		{
			throw std::runtime_error("Coordinate file does not contain triples of floats.");
		}

		const Eigen::Index rows = static_cast<Eigen::Index>(values.size() / 3);
		Matrix<float, Dynamic, Dynamic, RowMajor> coords(rows, 3);
		for (Eigen::Index i = 0; i < rows; ++i)
		{
			for (Eigen::Index j = 0; j < 3; ++j)
			{
				coords(i, j) = values[static_cast<size_t>(i * 3 + j)];
			}
		}
		return coords;
	}

	void expect_allclose(const Matrix<float, Dynamic, Dynamic, RowMajor> &actual,
	                     const Matrix<float, Dynamic, Dynamic, RowMajor> &expected,
	                     float rtol = 1e-4f,
	                     float atol = 1e-6f)
	{
		ASSERT_EQ(actual.rows(), expected.rows());
		ASSERT_EQ(actual.cols(), expected.cols());
		for (Eigen::Index i = 0; i < actual.rows(); ++i)
		{
			for (Eigen::Index j = 0; j < actual.cols(); ++j)
			{
				const float a = actual(i, j);
				const float b = expected(i, j);
				const float tol = atol + rtol * std::abs(b);
				EXPECT_LE(std::abs(a - b), tol);
			}
		}
	}

	template <typename DT>
	trxmmap::TrxFile<DT> *load_trx(const fs::path &path)
	{
		if (is_directory(path))
		{
			return trxmmap::load_from_directory<DT>(path.string());
		}
		return trxmmap::load_from_zip<DT>(path.string());
	}

	class ScopedEnvVar
	{
	public:
		ScopedEnvVar(const std::string &name, const std::string &value) : name_(name)
		{
			const char *existing = std::getenv(name_.c_str());
			if (existing != nullptr)
			{
				had_value_ = true;
				previous_ = existing;
			}
			set(value);
		}

		~ScopedEnvVar()
		{
			if (had_value_)
			{
				set(previous_);
			}
			else
			{
				unset();
			}
		}

	private:
		void set(const std::string &value)
		{
#if defined(_WIN32) || defined(_WIN64)
			_putenv_s(name_.c_str(), value.c_str());
#else
			setenv(name_.c_str(), value.c_str(), 1);
#endif
		}

		void unset()
		{
#if defined(_WIN32) || defined(_WIN64)
			_putenv_s(name_.c_str(), "");
#else
			unsetenv(name_.c_str());
#endif
		}

		std::string name_;
		bool had_value_ = false;
		std::string previous_;
	};

	std::string get_current_working_dir()
	{
		char buffer[PATH_MAX];
		if (getcwd(buffer, sizeof(buffer)) == nullptr)
		{
			throw std::runtime_error("Failed to get current working directory");
		}
		return std::string(buffer);
	}

	bool wait_for_path_gone(const fs::path &path, int retries = 10, int delay_ms = 50)
	{
		for (int i = 0; i < retries; ++i)
		{
			if (!fs::exists(path))
			{
				return true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
		}
		return !fs::exists(path);
	}
}

TEST(TrxFileIo, load_rasmm)
{
	const auto gs_dir = require_gold_standard_dir();
	const auto coords = load_rasmm_coords(gs_dir / "gs_rasmm_space.txt");

	const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
	for (const auto &input : inputs)
	{
		ASSERT_TRUE(fs::exists(input));
		trxmmap::TrxFile<float> *trx = load_trx<float>(input);
		Matrix<float, Dynamic, Dynamic, RowMajor> actual = trx->streamlines->_data;
		expect_allclose(actual, coords);
		trx->close();
		delete trx;
	}
}

TEST(TrxFileIo, multi_load_save_rasmm)
{
	const auto gs_dir = require_gold_standard_dir();
	const auto coords = load_rasmm_coords(gs_dir / "gs_rasmm_space.txt");

	const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
	for (const auto &input : inputs)
	{
		ASSERT_TRUE(fs::exists(input));
		fs::path tmp_dir = make_temp_test_dir("trx_gs");

		trxmmap::TrxFile<float> *trx = load_trx<float>(input);
		const std::string input_str = normalize_path(input.string());
		const std::string basename = trxmmap::get_base("/", input_str);
		const std::string ext = trxmmap::get_ext(input_str);
		const std::string basename_no_ext =
		    ext.empty() ? basename : basename.substr(0, basename.size() - ext.size() - 1);

		for (int i = 0; i < 3; ++i)
		{
			fs::path out_path = tmp_dir / (basename_no_ext + "_tmp" + std::to_string(i) +
			                               (ext.empty() ? "" : ("." + ext)));
			trxmmap::save(*trx, out_path.string());
			trx->close();
			delete trx;
			trx = load_trx<float>(out_path);
		}

		Matrix<float, Dynamic, Dynamic, RowMajor> actual = trx->streamlines->_data;
		expect_allclose(actual, coords);
		trx->close();
		delete trx;

		std::error_code ec;
		fs::remove_all(tmp_dir, ec);
	}
}

TEST(TrxFileIo, delete_tmp_gs_dir_rasmm)
{
	const auto gs_dir = require_gold_standard_dir();
	const auto coords = load_rasmm_coords(gs_dir / "gs_rasmm_space.txt");

	const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
	for (const auto &input : inputs)
	{
		ASSERT_TRUE(fs::exists(input));
		trxmmap::TrxFile<float> *trx = load_trx<float>(input);

		std::string tmp_dir = trx->_uncompressed_folder_handle;
		if (is_regular_file(input))
		{
			ASSERT_FALSE(tmp_dir.empty());
			ASSERT_TRUE(fs::exists(tmp_dir));
		}

		Matrix<float, Dynamic, Dynamic, RowMajor> actual = trx->streamlines->_data;
		expect_allclose(actual, coords);
		trx->close();

		if (is_regular_file(input))
		{
#if defined(_WIN32) || defined(_WIN64)
			// Windows can hold file handles briefly after close; avoid flaky removal assertions.
			(void)wait_for_path_gone(tmp_dir);
#else
			EXPECT_TRUE(wait_for_path_gone(tmp_dir));
#endif
		}

		delete trx;

		trx = load_trx<float>(input);
		Matrix<float, Dynamic, Dynamic, RowMajor> actual2 = trx->streamlines->_data;
		expect_allclose(actual2, coords);
		trx->close();
		delete trx;
	}
}

TEST(TrxFileIo, close_tmp_files)
{
	const auto gs_dir = require_gold_standard_dir();
	const fs::path input = gs_dir / "gs.trx";
	ASSERT_TRUE(fs::exists(input));

	trxmmap::TrxFile<float> *trx = load_trx<float>(input);
	const std::string tmp_dir = trx->_uncompressed_folder_handle;
	ASSERT_FALSE(tmp_dir.empty());
	ASSERT_TRUE(fs::exists(tmp_dir));

	const std::vector<fs::path> expected_paths = {
	    "offsets.uint32",
	    "positions.3.float32",
	    "header.json",
	    "dps/random_coord.3.float32",
	    "dpv/color_y.float32",
	    "dpv/color_x.float32",
	    "dpv/color_z.float32"};

	for (const auto &rel_path : expected_paths)
	{
		EXPECT_TRUE(fs::exists(fs::path(tmp_dir) / rel_path));
	}

	trx->close();
	delete trx;

#if defined(_WIN32) || defined(_WIN64)
	// Windows can hold file handles briefly after close; avoid flaky removal assertions.
	(void)wait_for_path_gone(tmp_dir);
#else
	EXPECT_TRUE(wait_for_path_gone(tmp_dir));
#endif
}

TEST(TrxFileIo, change_tmp_dir)
{
	const auto gs_dir = require_gold_standard_dir();
	const fs::path input = gs_dir / "gs.trx";
	ASSERT_TRUE(fs::exists(input));

	const char *home_env = std::getenv("HOME");
#if defined(_WIN32) || defined(_WIN64)
	if (home_env == nullptr || std::string(home_env).empty())
	{
		home_env = std::getenv("USERPROFILE");
	}
#endif
	if (home_env == nullptr || std::string(home_env).empty())
	{
		GTEST_SKIP() << "No HOME/USERPROFILE set for TRX_TMPDIR test";
	}

	{
		ScopedEnvVar env("TRX_TMPDIR", "use_working_dir");
		trxmmap::TrxFile<float> *trx = load_trx<float>(input);
		fs::path tmp_dir = trx->_uncompressed_folder_handle;
		fs::path parent = tmp_dir.parent_path();
		fs::path expected = fs::path(get_current_working_dir());
		std::string parent_norm = normalize_path(parent.lexically_normal());
		if (parent_norm == ".")
		{
			parent_norm = normalize_path(expected.lexically_normal());
		}
		EXPECT_EQ(parent_norm, normalize_path(expected.lexically_normal()));
		trx->close();
		delete trx;
	}

	{
		ScopedEnvVar env("TRX_TMPDIR", home_env);
		trxmmap::TrxFile<float> *trx = load_trx<float>(input);
		fs::path tmp_dir = trx->_uncompressed_folder_handle;
		fs::path parent = tmp_dir.parent_path();
		fs::path expected = fs::path(std::string(home_env));
		EXPECT_EQ(normalize_path(parent.lexically_normal()),
		          normalize_path(expected.lexically_normal()));
		trx->close();
		delete trx;
	}
}

TEST(TrxFileIo, complete_dir_from_trx)
{
	const auto gs_dir = require_gold_standard_dir();

	const std::set<std::string> expected_content = {
	    "offsets.uint32",
	    "positions.3.float32",
	    "header.json",
	    "dps/random_coord.3.float32",
	    "dpv/color_y.float32",
	    "dpv/color_x.float32",
	    "dpv/color_z.float32"};

	const std::vector<fs::path> inputs = {gs_dir / "gs.trx", gs_dir / "gs_fldr.trx"};
	for (const auto &input : inputs)
	{
		ASSERT_TRUE(fs::exists(input));
		trxmmap::TrxFile<float> *trx = load_trx<float>(input);
		fs::path dir_to_check = trx->_uncompressed_folder_handle.empty() ? input : trx->_uncompressed_folder_handle;

		std::set<std::string> file_paths;
		std::vector<fs::path> pending;
		pending.push_back(dir_to_check);
		while (!pending.empty())
		{
			fs::path current = pending.back();
			pending.pop_back();

			DIR *dir = opendir(current.c_str());
			if (dir == nullptr)
			{
				continue;
			}
			struct dirent *entry;
			while ((entry = readdir(dir)) != nullptr)
			{
				if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
				{
					continue;
				}
				fs::path child = current / entry->d_name;
				if (is_directory(child))
				{
					pending.push_back(child);
				}
				else if (is_regular_file(child))
				{
					std::string full = normalize_path(child);
					std::string base = normalize_path(dir_to_check);
					if (full.rfind(base, 0) == 0)
					{
						std::string rel = full.substr(base.size());
						if (!rel.empty() && rel[0] == '/')
						{
							rel.erase(0, 1);
						}
						file_paths.insert(rel);
					}
				}
			}
			closedir(dir);
		}

		EXPECT_EQ(file_paths, expected_content);
		trx->close();
		delete trx;
	}
}

TEST(TrxFileIo, complete_zip_from_trx)
{
	const auto gs_dir = require_gold_standard_dir();
	const fs::path input = gs_dir / "gs.trx";
	ASSERT_TRUE(fs::exists(input));

	const std::set<std::string> expected_content = {
	    "offsets.uint32",
	    "positions.3.float32",
	    "header.json",
	    "dps/random_coord.3.float32",
	    "dpv/color_y.float32",
	    "dpv/color_x.float32",
	    "dpv/color_z.float32"};

	int errorp = 0;
	zip_t *zf = zip_open(input.string().c_str(), 0, &errorp);
	ASSERT_NE(zf, nullptr);

	std::set<std::string> zip_file_list;
	zip_int64_t num_entries = zip_get_num_entries(zf, ZIP_FL_UNCHANGED);
	for (zip_int64_t i = 0; i < num_entries; ++i)
	{
		const char *entry_name = zip_get_name(zf, i, ZIP_FL_UNCHANGED);
		if (entry_name == nullptr)
		{
			continue;
		}
		std::string name(entry_name);
		if (!name.empty() && name.back() == '/')
		{
			continue;
		}
		zip_file_list.insert(name);
	}
	zip_close(zf);

	EXPECT_EQ(zip_file_list, expected_content);
}
