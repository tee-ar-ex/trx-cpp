#include <trx/legacy_io.h>

#include <exception>
#include <iostream>
#include <string>

#include <cxxopts.hpp>

namespace {
bool ends_with(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool load_tractogram(const std::string &input_file, trx::legacy::Tractogram &tractogram) {
  if (ends_with(input_file, ".trx")) {
    return trx::legacy::load_trx(input_file, tractogram);
  }
  if (ends_with(input_file, ".trk")) {
    return trx::legacy::load_trk(input_file, tractogram);
  }
  if (ends_with(input_file, ".tck")) {
    return trx::legacy::load_tck(input_file, tractogram);
  }
  if (ends_with(input_file, ".vtk")) {
    return trx::legacy::load_vtk(input_file, tractogram);
  }
  throw std::runtime_error("unsupported input format: " + input_file);
}

bool save_tractogram(const trx::legacy::Tractogram &tractogram,
                     const std::string &output_file,
                     const std::string &input_file,
                     const std::string &ref_path) {
  if (ends_with(output_file, ".trx")) {
    return trx::legacy::save_trx(tractogram, output_file, ref_path);
  }
  if (ends_with(output_file, ".trk")) {
    return trx::legacy::save_trk(tractogram, output_file, input_file, ref_path);
  }
  if (ends_with(output_file, ".tck")) {
    return trx::legacy::save_tck(tractogram, output_file);
  }
  if (ends_with(output_file, ".vtk")) {
    return trx::legacy::save_vtk(tractogram, output_file);
  }
  throw std::runtime_error("unsupported output format: " + output_file);
}

} // namespace

int main(int argc, char **argv) { // check_syntax off
  std::string help_text;
  try {
    std::string input_file;
    std::string output_file;
    std::string ref_path;
    cxxopts::Options options("trx-convert", "Convert tractograms between TRX and legacy formats (TRK/TCK/VTK).");
    options.add_options()("ref",
                          "Reference NIfTI file (required for TCK/VTK -> TRX/TRK)",
                          cxxopts::value<std::string>(ref_path)->default_value(""))(
        "input", "Input tractogram (.trx/.trk/.tck/.vtk)", cxxopts::value<std::string>(input_file))(
        "output", "Output tractogram (.trx/.trk/.tck/.vtk)", cxxopts::value<std::string>(output_file));
    options.parse_positional({"input", "output"});
    options.positional_help("<input> <output>");
    help_text = options.help();

    auto result = options.parse(argc, argv);
    if (result.count("input") == 0U || result.count("output") == 0U) {
      std::cerr << help_text << "\n";
      return 1;
    }
    input_file = result["input"].as<std::string>();
    output_file = result["output"].as<std::string>();
    ref_path = result["ref"].as<std::string>();

    // A reference is required only when going from a format that carries no spatial
    // header (TCK/VTK) to one that needs it (TRX/TRK).
    const bool input_lacks_reference = ends_with(input_file, ".tck") || ends_with(input_file, ".vtk");
    const bool output_needs_reference = ends_with(output_file, ".trx") || ends_with(output_file, ".trk");
    if (input_lacks_reference && output_needs_reference && ref_path.empty()) {
      std::cerr << "trx-convert: TCK/VTK -> TRX/TRK conversion requires --ref <nifti_file>\n";
      return 1;
    }

    trx::legacy::Tractogram tractogram;
    if (!load_tractogram(input_file, tractogram)) {
      std::cerr << "trx-convert: failed to load input file: " << input_file << "\n";
      return 1;
    }
    if (!save_tractogram(tractogram, output_file, input_file, ref_path)) {
      std::cerr << "trx-convert: failed to save output file: " << output_file << "\n";
      return 1;
    }
    return 0;
  } catch (const cxxopts::exceptions::exception &e) {
    std::cerr << "trx-convert: " << e.what() << "\n";
    if (!help_text.empty()) {
      std::cerr << help_text << "\n";
    }
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "trx-convert: " << e.what() << "\n";
    return 1;
  }
}
