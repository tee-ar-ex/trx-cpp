#ifndef TRX_FILESYSTEM_H
#define TRX_FILESYSTEM_H

#include <string>
#include <vector>
#include <system_error>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cctype>
#include <stdexcept>
#include <sys/stat.h>
#include <trx/dirent.h>
#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#include <windows.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace trx
{
  namespace fs
  {
    enum class perms : unsigned
    {
      none = 0,
      owner_read = 0400,
      owner_write = 0200,
      owner_exec = 0100,
      group_read = 0040,
      group_write = 0020,
      group_exec = 0010,
      others_read = 0004,
      others_write = 0002,
      others_exec = 0001,
      all = 0777
    };

    inline unsigned _perm_mask(perms p)
    {
      return static_cast<unsigned>(p);
    }
  } // namespace fs
} // namespace trx

namespace trx
{
  namespace fs
  {
    inline bool _is_separator(char c)
    {
      return c == '/' || c == '\\';
    }

    inline char _preferred_separator()
    {
#if defined(_WIN32) || defined(_WIN64)
      return '\\';
#else
      return '/';
#endif
    }

    class path
    {
    public:
      path() {}
      path(const std::string &p) : path_(p) {}
      path(const char *p) : path_(p ? p : "") {}

      std::string string() const { return path_; }
      const char *c_str() const { return path_.c_str(); }
      bool empty() const { return path_.empty(); }
      void clear() { path_.clear(); }

      bool is_absolute() const
      {
#if defined(_WIN32) || defined(_WIN64)
        if (path_.size() >= 2 && std::isalpha(path_[0]) && path_[1] == ':')
          return (path_.size() >= 3 && _is_separator(path_[2]));
        if (path_.size() >= 2 && _is_separator(path_[0]) && _is_separator(path_[1]))
          return true;
        return (!path_.empty() && _is_separator(path_[0]));
#else
        return (!path_.empty() && path_[0] == '/');
#endif
      }

      bool has_parent_path() const
      {
        return !parent_path().empty();
      }

      path parent_path() const
      {
        if (path_.empty())
          return path();

        size_t end = path_.size();
        while (end > 0 && _is_separator(path_[end - 1]))
          --end;
        if (end == 0)
          return path();

        size_t pos = path_.find_last_of("/\\", end - 1);
        if (pos == std::string::npos)
          return path();
        if (pos == 0)
          return path(std::string(1, _preferred_separator()));
#if defined(_WIN32) || defined(_WIN64)
        if (pos == 2 && std::isalpha(path_[0]) && path_[1] == ':')
          return path(path_.substr(0, 3));
#endif
        return path(path_.substr(0, pos));
      }

      path lexically_normal() const
      {
        if (path_.empty())
          return path();

        std::string root;
        size_t start = 0;
#if defined(_WIN32) || defined(_WIN64)
        if (path_.size() >= 2 && std::isalpha(path_[0]) && path_[1] == ':')
        {
          root = path_.substr(0, 2);
          start = 2;
          if (path_.size() >= 3 && _is_separator(path_[2]))
          {
            root += _preferred_separator();
            start = 3;
          }
        }
        else if (path_.size() >= 2 && _is_separator(path_[0]) && _is_separator(path_[1]))
        {
          root = std::string(2, _preferred_separator());
          start = 2;
        }
        else if (!path_.empty() && _is_separator(path_[0]))
        {
          root = std::string(1, _preferred_separator());
          start = 1;
        }
#else
        if (!path_.empty() && path_[0] == '/')
        {
          root = "/";
          start = 1;
        }
#endif

        std::vector<std::string> parts;
        size_t i = start;
        while (i < path_.size())
        {
          while (i < path_.size() && _is_separator(path_[i]))
            ++i;
          if (i >= path_.size())
            break;
          size_t j = i;
          while (j < path_.size() && !_is_separator(path_[j]))
            ++j;
          std::string part = path_.substr(i, j - i);
          if (part == ".")
          {
            // skip
          }
          else if (part == "..")
          {
            if (!parts.empty() && parts.back() != "..")
            {
              parts.pop_back();
            }
            else if (root.empty())
            {
              parts.push_back("..");
            }
          }
          else
          {
            parts.push_back(part);
          }
          i = j;
        }

        std::string out = root;
        for (size_t idx = 0; idx < parts.size(); ++idx)
        {
          if (!out.empty() && !_is_separator(out[out.size() - 1]))
            out += _preferred_separator();
          out += parts[idx];
        }

        if (out.empty() && root.empty())
          out = ".";

        return path(out);
      }

    private:
      std::string path_;
    };

    inline path operator/(const path &lhs, const path &rhs)
    {
      if (rhs.is_absolute())
        return rhs;
      if (lhs.string().empty())
        return rhs;
      std::string out = lhs.string();
      if (!_is_separator(out[out.size() - 1]))
        out += _preferred_separator();
      out += rhs.string();
      return path(out);
    }

    inline path operator/(const path &lhs, const char *rhs)
    {
      return lhs / path(rhs);
    }

    inline path operator/(const path &lhs, const std::string &rhs)
    {
      return lhs / path(rhs);
    }

    inline path operator/(const std::string &lhs, const path &rhs)
    {
      return path(lhs) / rhs;
    }

    inline path operator/(const char *lhs, const path &rhs)
    {
      return path(lhs) / rhs;
    }

    inline bool exists(const path &p, std::error_code &ec)
    {
      struct stat buf;
      if (stat(p.c_str(), &buf) == 0)
      {
        ec.clear();
        return true;
      }
      if (errno == ENOENT)
      {
        ec.clear();
        return false;
      }
      ec = std::error_code(errno, std::generic_category());
      return false;
    }

    inline bool exists(const path &p)
    {
      std::error_code ec;
      bool ok = exists(p, ec);
      if (ec)
        throw std::runtime_error(ec.message());
      return ok;
    }

    inline bool is_directory(const path &p, std::error_code &ec)
    {
      struct stat buf;
      if (stat(p.c_str(), &buf) == 0)
      {
        ec.clear();
        return S_ISDIR(buf.st_mode);
      }
      if (errno == ENOENT)
      {
        ec.clear();
        return false;
      }
      ec = std::error_code(errno, std::generic_category());
      return false;
    }

    inline bool create_directory(const path &p, std::error_code &ec)
    {
#if defined(_WIN32) || defined(_WIN64)
      int rc = _mkdir(p.c_str());
#else
      int rc = mkdir(p.c_str(), 0777);
#endif
      if (rc == 0)
      {
        ec.clear();
        return true;
      }
      if (errno == EEXIST)
      {
        ec.clear();
        return false;
      }
      ec = std::error_code(errno, std::generic_category());
      return false;
    }

    inline bool create_directories(const path &p, std::error_code &ec)
    {
      if (p.empty())
      {
        ec.clear();
        return false;
      }

      if (exists(p, ec))
      {
        if (ec)
          return false;
        if (is_directory(p, ec))
          return false;
        ec = std::make_error_code(std::errc::not_a_directory);
        return false;
      }

      path parent = p.parent_path();
      if (!parent.empty() && parent.string() != p.string())
      {
        create_directories(parent, ec);
        if (ec)
          return false;
      }

      return create_directory(p, ec);
    }

    inline path temp_directory_path(std::error_code &ec)
    {
      const char *candidates[] = {std::getenv("TMPDIR"), std::getenv("TEMP"), std::getenv("TMP")};
      for (const char *candidate : candidates)
      {
        if (!candidate || std::string(candidate).empty())
          continue;
        path p(candidate);
        if (is_directory(p, ec))
        {
          ec.clear();
          return p;
        }
        if (ec)
          ec.clear();
      }

#if defined(_WIN32) || defined(_WIN64)
      path fallback("C:\\Temp");
#else
      path fallback("/tmp");
#endif
      if (is_directory(fallback, ec))
      {
        ec.clear();
        return fallback;
      }
      if (!ec)
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
      return path();
    }

    inline std::uintmax_t remove_all(const path &p, std::error_code &ec)
    {
      struct stat buf;
      if (stat(p.c_str(), &buf) != 0)
      {
        if (errno == ENOENT)
        {
          ec.clear();
          return 0;
        }
        ec = std::error_code(errno, std::generic_category());
        return 0;
      }

      std::uintmax_t count = 0;
      if (S_ISDIR(buf.st_mode))
      {
        DIR *dir = opendir(p.c_str());
        if (!dir)
        {
          ec = std::error_code(errno, std::generic_category());
          return count;
        }
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL)
        {
          if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
          path child = path(p.string()) / entry->d_name;
          count += remove_all(child, ec);
          if (ec)
          {
            closedir(dir);
            return count;
          }
        }
        closedir(dir);
        if (rmdir(p.c_str()) != 0)
        {
          ec = std::error_code(errno, std::generic_category());
          return count;
        }
        ++count;
      }
      else
      {
        if (unlink(p.c_str()) != 0)
        {
          ec = std::error_code(errno, std::generic_category());
          return count;
        }
        ++count;
      }

      ec.clear();
      return count;
    }

    inline std::uintmax_t file_size(const path &p, std::error_code &ec)
    {
#if defined(_WIN32) || defined(_WIN64)
      struct _stat buf;
      if (_stat(p.c_str(), &buf) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return static_cast<std::uintmax_t>(-1);
      }
#else
      struct stat buf;
      if (stat(p.c_str(), &buf) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return static_cast<std::uintmax_t>(-1);
      }
#endif
      if (!S_ISREG(buf.st_mode))
      {
        if (S_ISDIR(buf.st_mode))
          ec = std::make_error_code(std::errc::is_a_directory);
        else
          ec = std::make_error_code(std::errc::invalid_argument);
        return static_cast<std::uintmax_t>(-1);
      }
      ec.clear();
      return static_cast<std::uintmax_t>(buf.st_size);
    }

    inline std::uintmax_t file_size(const path &p)
    {
      std::error_code ec;
      std::uintmax_t size = file_size(p, ec);
      if (ec)
        throw std::runtime_error(ec.message());
      return size;
    }

    inline bool is_symlink(const path &p, std::error_code &ec)
    {
#if defined(_WIN32) || defined(_WIN64)
      DWORD attrs = GetFileAttributesA(p.c_str());
      if (attrs == INVALID_FILE_ATTRIBUTES)
      {
        ec = std::error_code(GetLastError(), std::system_category());
        return false;
      }
      ec.clear();
      return (attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
#else
      struct stat buf;
      if (lstat(p.c_str(), &buf) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return false;
      }
      ec.clear();
      return S_ISLNK(buf.st_mode);
#endif
    }

    inline bool is_symlink(const path &p)
    {
      std::error_code ec;
      bool ok = is_symlink(p, ec);
      if (ec)
        throw std::runtime_error(ec.message());
      return ok;
    }

    inline bool create_symlink(const path &target, const path &link, std::error_code &ec)
    {
#if defined(_WIN32) || defined(_WIN64)
      DWORD flags = 0;
      struct _stat buf;
      if (_stat(target.c_str(), &buf) == 0 && (buf.st_mode & _S_IFDIR))
        flags |= SYMBOLIC_LINK_FLAG_DIRECTORY;
      if (CreateSymbolicLinkA(link.c_str(), target.c_str(), flags) == 0)
      {
        ec = std::error_code(GetLastError(), std::system_category());
        return false;
      }
      ec.clear();
      return true;
#else
      if (symlink(target.c_str(), link.c_str()) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return false;
      }
      ec.clear();
      return true;
#endif
    }

    inline void permissions(const path &p, perms prms, std::error_code &ec)
    {
#if defined(_WIN32) || defined(_WIN64)
      int mode = _S_IREAD;
      if ((_perm_mask(prms) & _perm_mask(perms::owner_write)) != 0)
        mode |= _S_IWRITE;
      if (_chmod(p.c_str(), mode) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return;
      }
      ec.clear();
#else
      mode_t mode = static_cast<mode_t>(_perm_mask(prms) & _perm_mask(perms::all));
      if (chmod(p.c_str(), mode) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return;
      }
      ec.clear();
#endif
    }

    inline perms permissions(const path &p, std::error_code &ec)
    {
#if defined(_WIN32) || defined(_WIN64)
      struct _stat buf;
      if (_stat(p.c_str(), &buf) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return perms::none;
      }
      ec.clear();
      return static_cast<perms>(buf.st_mode & 0777);
#else
      struct stat buf;
      if (stat(p.c_str(), &buf) != 0)
      {
        ec = std::error_code(errno, std::generic_category());
        return perms::none;
      }
      ec.clear();
      return static_cast<perms>(buf.st_mode & 0777);
#endif
    }

    inline perms permissions(const path &p)
    {
      std::error_code ec;
      perms out = permissions(p, ec);
      if (ec)
        throw std::runtime_error(ec.message());
      return out;
    }
  } // namespace fs
} // namespace trx

#endif
