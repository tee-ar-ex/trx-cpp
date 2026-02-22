#ifndef TRX_DETAIL_ZIP_RAII_H
#define TRX_DETAIL_ZIP_RAII_H

#include <string>
#include <zip.h>

#include <trx/detail/exceptions.h>

namespace trx {
namespace detail {

/// RAII wrapper for zip_t*. Calls zip_discard() on destruction unless commit() is called.
class ZipArchive {
public:
  ZipArchive() = default;
  explicit ZipArchive(zip_t *z) : z_(z) {}

  ~ZipArchive() {
    if (z_) {
      if (committed_) {
        zip_close(z_);
      } else {
        zip_discard(z_);
      }
    }
  }

  ZipArchive(const ZipArchive &) = delete;
  ZipArchive &operator=(const ZipArchive &) = delete;
  ZipArchive(ZipArchive &&o) noexcept : z_(o.z_), committed_(o.committed_) { o.z_ = nullptr; }
  ZipArchive &operator=(ZipArchive &&o) noexcept {
    if (this != &o) {
      if (z_)
        zip_discard(z_);
      z_ = o.z_;
      committed_ = o.committed_;
      o.z_ = nullptr;
    }
    return *this;
  }

  /// Commit changes (calls zip_close on destruction instead of zip_discard).
  void commit(const std::string &path = "") {
    committed_ = true;
    if (z_ && zip_close(z_) != 0) {
      auto err = zip_strerror(z_);
      z_ = nullptr; // prevent double-close
      throw TrxIOError("Unable to close archive " + path + ": " + err);
    }
    z_ = nullptr;
  }

  zip_t *get() const { return z_; }
  explicit operator bool() const { return z_ != nullptr; }

  /// Release ownership without closing.
  zip_t *release() {
    auto *p = z_;
    z_ = nullptr;
    return p;
  }

private:
  zip_t *z_ = nullptr;
  bool committed_ = false;
};

/// RAII wrapper for zip_file_t*. Calls zip_fclose() on destruction.
class ZipFile {
public:
  ZipFile() = default;
  explicit ZipFile(zip_file_t *f) : f_(f) {}

  ~ZipFile() {
    if (f_)
      zip_fclose(f_);
  }

  ZipFile(const ZipFile &) = delete;
  ZipFile &operator=(const ZipFile &) = delete;
  ZipFile(ZipFile &&o) noexcept : f_(o.f_) { o.f_ = nullptr; }
  ZipFile &operator=(ZipFile &&o) noexcept {
    if (this != &o) {
      if (f_)
        zip_fclose(f_);
      f_ = o.f_;
      o.f_ = nullptr;
    }
    return *this;
  }

  zip_file_t *get() const { return f_; }
  explicit operator bool() const { return f_ != nullptr; }

private:
  zip_file_t *f_ = nullptr;
};

} // namespace detail
} // namespace trx

#endif // TRX_DETAIL_ZIP_RAII_H
