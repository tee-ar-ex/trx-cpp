#include <iostream>
#include <string>
#include <trx/legacy_io.h>
#include <trx/trx.h>

int main(int argc, char** argv) {
    std::string input_file;
    std::string output_file;
    std::string ref_path;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ref") {
            if (i + 1 < argc) {
                ref_path = argv[++i];
            } else {
                std::cerr << "Error: --ref requires an argument\n";
                return 1;
            }
        } else if (input_file.empty()) {
            input_file = arg;
        } else if (output_file.empty()) {
            output_file = arg;
        }
    }
    
    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Usage: convert <input> <output> [--ref <nifti_file>]\n";
        return 1;
    }
    
    trx::legacy::Tractogram tr;
    bool success = false;
    
    auto ends_with = [](const std::string& str, const std::string& suffix) {
        return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    
    if (ends_with(input_file, ".trx")) success = trx::legacy::load_trx(input_file, tr);
    else if (ends_with(input_file, ".trk")) success = trx::legacy::load_trk(input_file, tr);
    else if (ends_with(input_file, ".tck")) success = trx::legacy::load_tck(input_file, tr);
    else if (ends_with(input_file, ".vtk")) success = trx::legacy::load_vtk(input_file, tr);
    
    if (!success) {
        std::cerr << "Error loading input file\n";
        return 1;
    }
    
    bool is_tck_vtk = ends_with(input_file, ".tck") || ends_with(input_file, ".vtk");
    bool is_trx_trk = ends_with(output_file, ".trx") || ends_with(output_file, ".trk");
    
    if (is_tck_vtk && is_trx_trk && ref_path.empty()) {
        std::cerr << "Error: TCK/VTK -> TRX/TRK conversion requires --ref <nifti_file>\n";
        return 1;
    }
    
    success = false;
    if (ends_with(output_file, ".trx")) success = trx::legacy::save_trx(tr, output_file, ref_path);
    else if (ends_with(output_file, ".trk")) success = trx::legacy::save_trk(tr, output_file, input_file, ref_path);
    else if (ends_with(output_file, ".tck")) success = trx::legacy::save_tck(tr, output_file);
    else if (ends_with(output_file, ".vtk")) success = trx::legacy::save_vtk(tr, output_file);
    
    if (!success) {
        std::cerr << "Error saving output file\n";
        return 1;
    }
    return 0;
}
