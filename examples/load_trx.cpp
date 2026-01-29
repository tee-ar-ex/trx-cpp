#include "../src/trx.h"

using namespace trxmmap;
int main(int argc, char **argv) { // check_syntax off
  trxmmap::TrxFile<half> *trx = trxmmap::load_from_zip<half>(argv[1]);

  std::cout << "Vertices: " << trx->streamlines->_data.size() / 3 << "\n";
  std::cout << "First vertex (x,y,z): " << trx->streamlines->_data(0, 0) << "," << trx->streamlines->_data(0, 1) << ","
            << trx->streamlines->_data(0, 2) << "\n";
  std::cout << "Streamlines: " << trx->streamlines->_offsets.size() << "\n";
  std::cout << "Vertices in first streamline: " << trx->streamlines->_offsets(1) - trx->streamlines->_offsets(0)
            << "\n";
  std::cout << "dpg (data_per_group) items: " << trx->data_per_group.size() << "\n";
  std::cout << "dps (data_per_streamline) items: " << trx->data_per_streamline.size() << "\n";

  for (auto const &x : trx->data_per_streamline) {
    std::cout << "'" << x.first << "' items: " << x.second->_matrix.size() << "\n";
  }

  std::cout << "dpv (data_per_vertex) items:" << trx->data_per_vertex.size() << "\n";
  for (auto const &x : trx->data_per_vertex) {
    std::cout << "'" << x.first << "' items: " << x.second->_data.size() << "\n";
  }

  std::cout << *trx << std::endl;
}