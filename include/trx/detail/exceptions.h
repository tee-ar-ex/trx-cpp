#ifndef TRX_DETAIL_EXCEPTIONS_H
#define TRX_DETAIL_EXCEPTIONS_H

#include <stdexcept>
#include <string>

namespace trx {

/// Base exception for all TRX library errors.
class TrxError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// I/O errors: zip failures, file not found, mmap errors, write failures.
class TrxIOError : public TrxError {
public:
  using TrxError::TrxError;
};

/// Format errors: wrong sizes, missing fields, corrupt data, structural issues.
class TrxFormatError : public TrxError {
public:
  using TrxError::TrxError;
};

/// Dtype errors: unsupported or mismatched data types.
class TrxDTypeError : public TrxError {
public:
  using TrxError::TrxError;
};

/// Argument errors: invalid API arguments.
class TrxArgumentError : public TrxError {
public:
  using TrxError::TrxError;
};

} // namespace trx

#endif // TRX_DETAIL_EXCEPTIONS_H
