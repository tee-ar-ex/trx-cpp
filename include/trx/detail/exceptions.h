#ifndef TRX_DETAIL_EXCEPTIONS_H
#define TRX_DETAIL_EXCEPTIONS_H

#include <stdexcept>
#include <string>

#include <trx/trx_export.h>

namespace trx {

/// Base exception for all TRX library errors.
class TRX_EXPORT TrxError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// I/O errors: zip failures, file not found, mmap errors, write failures.
class TRX_EXPORT TrxIOError : public TrxError {
public:
  using TrxError::TrxError;
};

/// Format errors: wrong sizes, missing fields, corrupt data, structural issues.
class TRX_EXPORT TrxFormatError : public TrxError {
public:
  using TrxError::TrxError;
};

/// Dtype errors: unsupported or mismatched data types.
class TRX_EXPORT TrxDTypeError : public TrxError {
public:
  using TrxError::TrxError;
};

/// Argument errors: invalid API arguments.
class TRX_EXPORT TrxArgumentError : public TrxError {
public:
  using TrxError::TrxError;
};

} // namespace trx

#endif // TRX_DETAIL_EXCEPTIONS_H
