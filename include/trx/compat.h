#ifndef TRX_COMPAT_H
#define TRX_COMPAT_H

#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <direct.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <trx/dirent.h>

#ifndef S_IRWXU
#define S_IRWXU (_S_IREAD | _S_IWRITE)
#endif

inline int trx_mkdir(const char *path, int)
{
	return _mkdir(path);
}

inline int trx_unlink(const char *path)
{
	return _unlink(path);
}

inline int trx_rmdir(const char *path)
{
	return _rmdir(path);
}

#define mkdir(path, mode) trx_mkdir(path, mode)
#define unlink(path) trx_unlink(path)
#define rmdir(path) trx_rmdir(path)
#define open _open
#define read _read
#define write _write
#define close _close

#else
#include <dirent.h>
#endif

#endif
