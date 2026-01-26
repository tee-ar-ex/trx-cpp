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

inline int trx_open(const char *path, int oflag, int pmode = 0)
{
	return _open(path, oflag, pmode);
}

inline int trx_read(int fd, void *buffer, unsigned int count)
{
	return _read(fd, buffer, count);
}

inline int trx_write(int fd, const void *buffer, unsigned int count)
{
	return _write(fd, buffer, count);
}

inline int trx_close(int fd)
{
	return _close(fd);
}

#define mkdir(path, mode) trx_mkdir(path, mode)
#define unlink(path) trx_unlink(path)
#define rmdir(path) trx_rmdir(path)

#else
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

inline int trx_open(const char *path, int oflag, int pmode = 0)
{
	return ::open(path, oflag, pmode);
}

inline int trx_read(int fd, void *buffer, unsigned int count)
{
	return ::read(fd, buffer, count);
}

inline int trx_write(int fd, const void *buffer, unsigned int count)
{
	return ::write(fd, buffer, count);
}

inline int trx_close(int fd)
{
	return ::close(fd);
}
#endif

#endif
