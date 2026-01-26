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

#if defined(TRX_USE_BOOST_FILESYSTEM)
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

namespace trx
{
  namespace fs
  {
    using boost::filesystem::path;

    inline void _assign_error(std::error_code &out, const boost::system::error_code &in)
    {
      if (in)
        out = std::error_code(in.value(), std::generic_category());
      else
        out.clear();
    }

    inline bool exists(const path &p, std::error_code &ec)
    {
      boost::system::error_code bec;
      bool ok = boost::filesystem::exists(p, bec);
      _assign_error(ec, bec);
      return ok;
    }

    inline bool exists(const path &p)
    {
      boost::system::error_code bec;
      bool ok = boost::filesystem::exists(p, bec);
      if (bec)
        throw std::runtime_error(bec.message());
      return ok;
    }

    inline bool is_directory(const path &p, std::error_code &ec)
    {
      boost::system::error_code bec;
      bool ok = boost::filesystem::is_directory(p, bec);
      _assign_error(ec, bec);
      return ok;
    }

    inline bool create_directory(const path &p, std::error_code &ec)
    {
      boost::system::error_code bec;
      bool ok = boost::filesystem::create_directory(p, bec);
      _assign_error(ec, bec);
      return ok;
    }

    inline bool create_directories(const path &p, std::error_code &ec)
    {
      boost::system::error_code bec;
      bool ok = boost::filesystem::create_directories(p, bec);
      _assign_error(ec, bec);
      return ok;
    }

    inline path temp_directory_path(std::error_code &ec)
    {
      boost::system::error_code bec;
      path p = boost::filesystem::temp_directory_path(bec);
      _assign_error(ec, bec);
      return p;
    }

    inline std::uintmax_t remove_all(const path &p, std::error_code &ec)
    {
      boost::system::error_code bec;
      std::uintmax_t count = boost::filesystem::remove_all(p, bec);
      _assign_error(ec, bec);
      return count;
    }
  } // namespace fs
} // namespace trx

#else

#include <sys/stat.h>
#include <trx/dirent.h>
#include <unistd.h>
#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#endif

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
  } // namespace fs
} // namespace trx

#endif

#endif
