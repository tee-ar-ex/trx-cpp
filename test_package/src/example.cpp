#include <trx/trx.h>

int main()
{
    // Basic construction and cleanup exercises the public API and linkage.
    trxmmap::TrxFile<float> file;
    file.close();
    return 0;
}
