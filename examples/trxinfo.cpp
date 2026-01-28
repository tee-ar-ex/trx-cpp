#include <trx/trx.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "cli_colors.h"

namespace
{
std::string format_json_array(const json &value)
{
	if (!value.is_array())
	{
		return "n/a";
	}
	std::ostringstream out;
	out << "[";
	const auto items = value.array_items();
	for (size_t i = 0; i < items.size(); ++i)
	{
		if (i > 0)
		{
			out << ", ";
		}
		if (items[i].is_number())
		{
			out << std::fixed << std::setprecision(4) << items[i].number_value();
		}
		else
		{
			out << items[i].dump();
		}
	}
	out << "]";
	return out.str();
}

std::string format_matrix_row(const json &row)
{
	if (!row.is_array())
	{
		return row.dump();
	}
	std::ostringstream out;
	out << "[";
	const auto items = row.array_items();
	for (size_t i = 0; i < items.size(); ++i)
	{
		if (i > 0)
		{
			out << ", ";
		}
		out << std::fixed << std::setprecision(4) << items[i].number_value();
	}
	out << "]";
	return out.str();
}

template <typename DT>
void print_trx_info(trxmmap::TrxFile<DT> *trx,
                    const std::string &path,
                    bool is_dir,
                    trxmmap::TrxScalarType dtype,
                    bool show_stats)
{
	trx_cli::Colors colors;
	colors.enabled = trx_cli::stdout_supports_color();

	std::cout << trx_cli::colorize(colors, colors.bold, "TRX info") << "\n";
	std::cout << trx_cli::colorize(colors, colors.cyan, "Path") << ": " << path << "\n";
	std::cout << trx_cli::colorize(colors, colors.cyan, "Storage") << ": " << (is_dir ? "directory" : "zip archive") << "\n";
	std::cout << trx_cli::colorize(colors, colors.cyan, "Positions dtype") << ": "
	          << trxmmap::scalar_type_name(dtype) << "\n";

	const json &header = trx->header;
	if (!header.is_null())
	{
		std::cout << trx_cli::colorize(colors, colors.green, "Header") << ":\n";
		const json &nb_streamlines = header["NB_STREAMLINES"];
		const json &nb_vertices = header["NB_VERTICES"];
		const json &dimensions = header["DIMENSIONS"];
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Streamlines")
		          << ": " << (nb_streamlines.is_number() ? std::to_string(nb_streamlines.int_value()) : "n/a") << "\n";
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Vertices")
		          << ": " << (nb_vertices.is_number() ? std::to_string(nb_vertices.int_value()) : "n/a") << "\n";
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Dimensions") << ": " << format_json_array(dimensions) << "\n";

		const json &voxel_to_rasmm = header["VOXEL_TO_RASMM"];
		if (voxel_to_rasmm.is_array() && voxel_to_rasmm.array_items().size() == 4)
		{
			std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Voxel->RASMM") << ":\n";
			for (const auto &row : voxel_to_rasmm.array_items())
			{
				std::cout << "    " << format_matrix_row(row) << "\n";
			}
		}
	}

	if (show_stats && trx->streamlines != nullptr)
	{
		const auto &lengths = trx->streamlines->_lengths;
		const Eigen::Index count = lengths.size();
		if (count > 0)
		{
			uint64_t min_len = lengths(0);
			uint64_t max_len = lengths(0);
			uint64_t total_len = 0;
			for (Eigen::Index i = 0; i < count; ++i)
			{
				const uint64_t value = lengths(i);
				min_len = std::min<uint64_t>(min_len, value);
				max_len = std::max<uint64_t>(max_len, value);
				total_len += value;
			}
			const double mean_len = static_cast<double>(total_len) / static_cast<double>(count);
			std::cout << trx_cli::colorize(colors, colors.green, "Streamline lengths") << ":\n";
			std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Count") << ": " << count << "\n";
			std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Min/Mean/Max")
			          << ": " << min_len << " / " << std::fixed << std::setprecision(2) << mean_len
			          << " / " << max_len << "\n";
		}
	}

	std::cout << trx_cli::colorize(colors, colors.green, "Data arrays") << ":\n";
	if (!trx->data_per_vertex.empty())
	{
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Data per vertex") << ": " << trx->data_per_vertex.size() << "\n";
		for (const auto &kv : trx->data_per_vertex)
		{
			const auto &arr = kv.second->_data;
			std::cout << "    - " << kv.first << " (" << arr.rows() << "x" << arr.cols() << ")\n";
		}
	}
	else
	{
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Data per vertex") << ": none\n";
	}

	if (!trx->data_per_streamline.empty())
	{
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Data per streamline") << ": " << trx->data_per_streamline.size() << "\n";
		for (const auto &kv : trx->data_per_streamline)
		{
			const auto &arr = kv.second->_matrix;
			std::cout << "    - " << kv.first << " (" << arr.rows() << "x" << arr.cols() << ")\n";
		}
	}
	else
	{
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Data per streamline") << ": none\n";
	}

	if (!trx->groups.empty())
	{
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Groups") << ": " << trx->groups.size() << "\n";
		for (const auto &kv : trx->groups)
		{
			const auto &arr = kv.second->_matrix;
			std::cout << "    - " << kv.first << " (" << arr.rows() << "x" << arr.cols() << ")\n";
		}
	}
	else
	{
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Groups") << ": none\n";
	}

	if (!trx->data_per_group.empty())
	{
		std::cout << "  " << trx_cli::colorize(colors, colors.cyan, "Data per group") << ":\n";
		for (const auto &grp : trx->data_per_group)
		{
			std::cout << "    - " << trx_cli::colorize(colors, colors.magenta, grp.first) << ": " << grp.second.size() << "\n";
			for (const auto &kv : grp.second)
			{
				const auto &arr = kv.second->_matrix;
				std::cout << "      * " << kv.first << " (" << arr.rows() << "x" << arr.cols() << ")\n";
			}
		}
	}
	else
	{
		std::cout << "  " << colorize(colors, colors.cyan, "Data per group") << ": none\n";
	}
}

struct ReaderPrinter
{
	const std::string &path;
	bool is_dir;
	bool show_stats;

	template <typename ReaderT>
	int operator()(ReaderT &reader, trxmmap::TrxScalarType dtype) const
	{
		print_trx_info(reader.get(), path, is_dir, dtype, show_stats);
		return 0;
	}
};

} // namespace

int main(int argc, char **argv)
{
	CLI::App app{"Print information about a TRX file or directory."};
	std::string path;
	bool show_stats = false;

	app.add_option("path", path, "Path to TRX file or directory")->required();
	app.add_flag("--stats", show_stats, "Compute Min/Mean/Max streamline lengths");

	CLI11_PARSE(app, argc, argv);
	try
	{
		const bool is_dir = trxmmap::is_trx_directory(path);
		ReaderPrinter printer{path, is_dir, show_stats};
		return trxmmap::with_trx_reader(path, printer);
	}
	catch (const std::exception &e)
	{
		std::cerr << "trxinfo: " << e.what() << "\n";
		return 1;
	}
}
