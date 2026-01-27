#include <gtest/gtest.h>
#include <trx/filesystem.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <string>

namespace fs = trx::fs;

namespace
{
	std::string normalize_path(const std::string &path)
	{
		std::string out = path;
		std::replace(out.begin(), out.end(), '\\', '/');
		return out;
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

	void write_text_file(const fs::path &path, const std::string &content)
	{
		std::ofstream out(path.string());
		if (!out.is_open())
		{
			throw std::runtime_error("Failed to write test file");
		}
		out << content;
		out.close();
	}
}

// Validates path joining and lexical normalization.
TEST(TrxFilesystem, PathBasics)
{
	fs::path base("/tmp");
	fs::path child = base / "trx" / "file.txt";
	EXPECT_FALSE(child.empty());
	EXPECT_EQ(normalize_path(child.parent_path().lexically_normal().string()), "/tmp/trx");

	fs::path normalized = fs::path("/tmp/../tmp/./trx//file.txt").lexically_normal();
	EXPECT_EQ(normalize_path(normalized.string()), "/tmp/trx/file.txt");
}

// Exercises exists/is_directory/create_directories behavior.
TEST(TrxFilesystem, ExistsAndDirectories)
{
	fs::path root = make_temp_test_dir("trx_fs");
	fs::path nested = root / "a" / "b" / "c";
	std::error_code ec;

	EXPECT_FALSE(fs::exists(nested, ec));
	EXPECT_FALSE(ec);

	EXPECT_TRUE(fs::create_directories(nested, ec));
	EXPECT_FALSE(ec);
	EXPECT_TRUE(fs::exists(nested, ec));
	EXPECT_FALSE(ec);
	EXPECT_TRUE(fs::is_directory(nested, ec));
	EXPECT_FALSE(ec);

	fs::remove_all(root, ec);
	EXPECT_FALSE(ec);
}

// Checks file_size for files and error on directories; removes trees.
TEST(TrxFilesystem, FileSizeAndRemoveAll)
{
	fs::path root = make_temp_test_dir("trx_fs_size");
	fs::path file = root / "file.txt";
	write_text_file(file, "abc");

	EXPECT_EQ(fs::file_size(file), 3u);

	EXPECT_THROW(fs::file_size(root), std::runtime_error);

	std::error_code ec;
	fs::remove_all(root, ec);
	EXPECT_FALSE(ec);
	EXPECT_FALSE(fs::exists(root, ec));
}

// Verifies permissions API round-trip (skips if unsupported).
TEST(TrxFilesystem, PermissionsRoundTrip)
{
	fs::path root = make_temp_test_dir("trx_fs_perm");
	fs::path file = root / "perm.txt";
	write_text_file(file, "perm");

	std::error_code ec;
	auto perms_before = fs::permissions(file, ec);
	if (ec)
	{
		GTEST_SKIP() << "permissions() not supported on this platform";
	}

	fs::permissions(file, fs::perms::owner_read, ec);
	if (ec)
	{
		GTEST_SKIP() << "permissions() failed on this platform";
	}

	auto perms_after = fs::permissions(file, ec);
	if (!ec)
	{
		EXPECT_NE(static_cast<unsigned>(perms_after), 0u);
	}

	fs::permissions(file, perms_before, ec);

	fs::remove_all(root, ec);
}

// Creates and detects symlinks when permitted by the OS.
TEST(TrxFilesystem, SymlinkCreateAndDetect)
{
	fs::path root = make_temp_test_dir("trx_fs_link");
	fs::path target = root / "target.txt";
	fs::path link = root / "link.txt";
	write_text_file(target, "data");

	std::error_code ec;
	if (!fs::create_symlink(target, link, ec))
	{
		GTEST_SKIP() << "Symlink creation not permitted: " << ec.message();
	}

	EXPECT_TRUE(fs::exists(link, ec));
	EXPECT_FALSE(ec);
	EXPECT_TRUE(fs::is_symlink(link, ec));
	EXPECT_FALSE(ec);

	fs::remove_all(root, ec);
}
