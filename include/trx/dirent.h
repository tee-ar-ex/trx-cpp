#ifndef TRX_DIRENT_H
#define TRX_DIRENT_H

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <string>
#include <cstring>

#ifndef DT_UNKNOWN
#define DT_UNKNOWN 0
#endif
#ifndef DT_DIR
#define DT_DIR 4
#endif
#ifndef DT_REG
#define DT_REG 8
#endif

struct dirent
{
	char d_name[MAX_PATH];
	unsigned char d_type;
};

struct DIR
{
	HANDLE handle;
	WIN32_FIND_DATAA data;
	bool first;
	std::string pattern;
	dirent entry;
};

inline DIR *opendir(const char *name)
{
	if (!name || !*name)
		return nullptr;
	DIR *dir = new DIR();
	dir->first = true;
	dir->pattern = std::string(name);
	if (!dir->pattern.empty() && dir->pattern.back() != '\\' && dir->pattern.back() != '/')
		dir->pattern += "\\*";
	else
		dir->pattern += "*";
	dir->handle = FindFirstFileA(dir->pattern.c_str(), &dir->data);
	if (dir->handle == INVALID_HANDLE_VALUE)
	{
		delete dir;
		return nullptr;
	}
	return dir;
}

inline dirent *readdir(DIR *dirp)
{
	if (!dirp)
		return nullptr;
	if (dirp->first)
	{
		dirp->first = false;
	}
	else
	{
		if (!FindNextFileA(dirp->handle, &dirp->data))
			return nullptr;
	}
	const char *name = dirp->data.cFileName;
	strncpy_s(dirp->entry.d_name, name, MAX_PATH - 1);
	dirp->entry.d_name[MAX_PATH - 1] = '\0';
	dirp->entry.d_type = (dirp->data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? DT_DIR : DT_REG;
	return &dirp->entry;
}

inline int closedir(DIR *dirp)
{
	if (!dirp)
		return -1;
	if (dirp->handle != INVALID_HANDLE_VALUE)
		FindClose(dirp->handle);
	delete dirp;
	return 0;
}

#else
#include <dirent.h>
#endif

#endif
