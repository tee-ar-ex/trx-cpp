#pragma once

// Provide the public include path expected by consumers.
// This wrapper forwards to the actual header in src/ while
// also ensuring SPDLOG uses the external fmt if provided.
#ifndef SPDLOG_FMT_EXTERNAL
#define SPDLOG_FMT_EXTERNAL
#endif

#include "../../src/trx.h"
