#ifndef TRX_EXAMPLES_CLI_COLORS_H
#define TRX_EXAMPLES_CLI_COLORS_H

#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace trx_cli
{
struct Colors
{
	bool enabled = false;
	const char *reset = "\033[0m";
	const char *bold = "\033[1m";
	const char *cyan = "\033[36m";
	const char *green = "\033[32m";
	const char *yellow = "\033[33m";
	const char *magenta = "\033[35m";
};

inline bool stdout_supports_color()
{
#if defined(_WIN32) || defined(_WIN64)
	HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
	if (h == INVALID_HANDLE_VALUE || h == nullptr)
	{
		return false;
	}
	DWORD mode = 0;
	return GetConsoleMode(h, &mode) != 0;
#else
	return isatty(fileno(stdout)) != 0;
#endif
}

inline std::string colorize(const Colors &colors, const char *code, const std::string &text)
{
	if (!colors.enabled)
	{
		return text;
	}
	return std::string(code) + text + colors.reset;
}
} // namespace trx_cli

#endif
