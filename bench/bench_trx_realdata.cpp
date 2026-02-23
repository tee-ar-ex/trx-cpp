// Benchmark TRX streaming workloads for realistic datasets.
#include <benchmark/benchmark.h>
#include <trx/trx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
using Eigen::half;

std::string g_reference_trx_path;
bool g_reference_has_dpv = false;
size_t g_reference_streamline_count = 0;

constexpr float kMinLengthMm = 20.0f;
constexpr float kMaxLengthMm = 500.0f;
constexpr float kStepMm = 2.0f;
constexpr float kSlabThicknessMm = 5.0f;
constexpr size_t kSlabCount = 20;

constexpr std::array<size_t, 5> kStreamlineCounts = {100000, 500000, 1000000, 5000000, 10000000};

struct Fov {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float min_z;
  float max_z;
};

constexpr Fov kFov{-70.0f, 70.0f, -108.0f, 79.0f, -60.0f, 75.0f};

enum class GroupScenario : int { None = 0, Bundles = 1, Connectome = 2 };

constexpr size_t kBundleCount = 80;
constexpr std::array<size_t, 3> kConnectomeAtlasSizes = {80, 400, 1000};
constexpr size_t kConnectomeTotalGroups = 1480;  // sum of atlas sizes

std::string make_temp_path(const std::string &prefix) {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
  const auto dir = std::filesystem::temp_directory_path();
  return (dir / (prefix + "_" + std::to_string(id) + ".trx")).string();
}

std::string make_work_dir_name(const std::string &prefix) {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
#if defined(__unix__) || defined(__APPLE__)
  const auto pid = static_cast<uint64_t>(getpid());
#else
  const auto pid = static_cast<uint64_t>(0);
#endif
  const auto dir = std::filesystem::current_path();
  return (dir / (prefix + "_" + std::to_string(pid) + "_" + std::to_string(id))).string();
}

void register_cleanup(const std::string &path);

size_t file_size_bytes(const std::string &path) {
  std::error_code ec;
  if (!trx::fs::exists(path, ec)) {
    return 0;
  }
  if (trx::fs::is_directory(path, ec)) {
    size_t total = 0;
    for (trx::fs::recursive_directory_iterator it(path, ec), end; it != end; it.increment(ec)) {
      if (ec) {
        break;
      }
      if (!it->is_regular_file(ec)) {
        continue;
      }
      total += static_cast<size_t>(trx::fs::file_size(it->path(), ec));
      if (ec) {
        break;
      }
    }
    return total;
  }
  return static_cast<size_t>(trx::fs::file_size(path, ec));
}

double get_max_rss_kb() {
#if defined(__unix__) || defined(__APPLE__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0.0;
  }
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
#else
  return static_cast<double>(usage.ru_maxrss);
#endif
#else
  return 0.0;
#endif
}

size_t parse_env_size(const char *name, size_t default_value) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw) {
    return default_value;
  }
  return static_cast<size_t>(value);
}

bool parse_env_bool(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return default_value;
  }
  return std::string(raw) != "0";
}

int parse_env_int(const char *name, int default_value) {
  const char *raw = std::getenv(name);
  if (!raw || raw[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw) {
    return default_value;
  }
  return static_cast<int>(value);
}

bool is_core_profile() {
  const char *raw = std::getenv("TRX_BENCH_PROFILE");
  return raw && std::string(raw) == "core";
}

bool include_bundles_in_core_profile() {
  return parse_env_bool("TRX_BENCH_CORE_INCLUDE_BUNDLES", false);
}

size_t core_dpv_max_streamlines() {
  return parse_env_size("TRX_BENCH_CORE_DPV_MAX_STREAMLINES", 1000000);
}

size_t core_zip_max_streamlines() {
  return parse_env_size("TRX_BENCH_CORE_ZIP_MAX_STREAMLINES", 1000000);
}

std::vector<int> group_cases_for_benchmarks() {
  std::vector<int> groups = {static_cast<int>(GroupScenario::None)};
  if (!is_core_profile() || include_bundles_in_core_profile()) {
    groups.push_back(static_cast<int>(GroupScenario::Bundles));
  }
  if (parse_env_bool("TRX_BENCH_INCLUDE_CONNECTOME", !is_core_profile())) {
    groups.push_back(static_cast<int>(GroupScenario::Connectome));
  }
  return groups;
}

size_t group_count_for(GroupScenario scenario) {
  switch (scenario) {
  case GroupScenario::Bundles:
    return kBundleCount;
  case GroupScenario::Connectome:
    return kConnectomeTotalGroups;
  case GroupScenario::None:
  default:
    return 0;
  }
}

// Compute position buffer size based on streamline count.
// For slow storage (spinning disks, network filesystems), set TRX_BENCH_BUFFER_MULTIPLIER
// to 2-8 to reduce I/O frequency at the cost of higher memory usage.
// Example: multiplier=4 scales 256 MB → 1 GB for 1M streamlines.
std::size_t buffer_bytes_for_streamlines(std::size_t streamlines) {
  std::size_t base_bytes;
  if (streamlines >= 5000000) {
    base_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2 GB
  } else if (streamlines >= 1000000) {
    base_bytes = 256ULL * 1024ULL * 1024ULL;  // 256 MB
  } else {
    base_bytes = 16ULL * 1024ULL * 1024ULL;  // 16 MB
  }
  
  // Allow scaling buffer sizes for slower storage (HDD, NFS) to amortize I/O latency
  const size_t multiplier = std::max<size_t>(1, parse_env_size("TRX_BENCH_BUFFER_MULTIPLIER", 1));
  return base_bytes * multiplier;
}

std::vector<size_t> streamlines_for_benchmarks() {
  const size_t only = parse_env_size("TRX_BENCH_ONLY_STREAMLINES", 0);
  if (only > 0) {
    return {only};
  }
  const size_t max_val = parse_env_size("TRX_BENCH_MAX_STREAMLINES", 10000000);
  std::vector<size_t> counts = {10000000, 5000000, 1000000, 500000, 100000};
  counts.erase(std::remove_if(counts.begin(), counts.end(), [&](size_t v) { return v > max_val; }), counts.end());
  if (counts.empty()) {
    counts.push_back(max_val);
  }
  return counts;
}

void log_bench_start(const std::string &name, const std::string &details) {
  if (!parse_env_bool("TRX_BENCH_LOG", false)) {
    return;
  }
  std::cerr << "[trx-bench] start " << name << " " << details << std::endl;
}

void log_bench_end(const std::string &name, const std::string &details) {
  if (!parse_env_bool("TRX_BENCH_LOG", false)) {
    return;
  }
  std::cerr << "[trx-bench] end " << name << " " << details << std::endl;
}

const std::vector<std::string> &group_names_for(GroupScenario scenario) {
  static const std::vector<std::string> empty;
  static const std::vector<std::string> bundle_names = []() {
    std::vector<std::string> names;
    names.reserve(kBundleCount);
    for (size_t i = 1; i <= kBundleCount; ++i) {
      names.push_back("Bundle" + std::to_string(i));
    }
    return names;
  }();
  static const std::vector<std::string> connectome_names = []() {
    std::vector<std::string> names;
    names.reserve(kConnectomeTotalGroups);
    for (size_t a = 0; a < kConnectomeAtlasSizes.size(); ++a) {
      for (size_t r = 1; r <= kConnectomeAtlasSizes[a]; ++r) {
        names.push_back("atlas" + std::to_string(a + 1) + "_region" + std::to_string(r));
      }
    }
    return names;
  }();
  switch (scenario) {
  case GroupScenario::Bundles:
    return bundle_names;
  case GroupScenario::Connectome:
    return connectome_names;
  case GroupScenario::None:
  default:
    return empty;
  }
}

std::vector<uint32_t> build_prefix_ids(size_t num_streamlines) {
  if (num_streamlines > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("Too many streamlines for uint32 index space.");
  }
  std::vector<uint32_t> ids;
  ids.reserve(num_streamlines);
  for (size_t i = 0; i < num_streamlines; ++i) {
    ids.push_back(static_cast<uint32_t>(i));
  }
  return ids;
}

void assign_groups_to_trx(trx::TrxFile<half> &trx, GroupScenario scenario, size_t streamlines) {
  const auto group_count = group_count_for(scenario);
  const auto &group_names = group_names_for(scenario);
  if (group_count == 0) {
    return;
  }

  if (scenario == GroupScenario::Connectome) {
    std::vector<std::vector<uint32_t>> group_indices(kConnectomeTotalGroups);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    for (size_t i = 0; i < streamlines; ++i) {
      size_t group_offset = 0;
      for (size_t a = 0; a < kConnectomeAtlasSizes.size(); ++a) {
        const size_t n_regions = kConnectomeAtlasSizes[a];
        const size_t picks = (coin(rng) < 0.1f) ? 3 : 2;
        std::unordered_set<size_t> chosen;
        std::uniform_int_distribution<size_t> region_dist(0, n_regions - 1);
        while (chosen.size() < picks) {
          chosen.insert(region_dist(rng));
        }
        for (size_t r : chosen) {
          group_indices[group_offset + r].push_back(static_cast<uint32_t>(i));
        }
        group_offset += n_regions;
      }
    }
    for (size_t g = 0; g < kConnectomeTotalGroups; ++g) {
      trx.add_group_from_indices(group_names[g], group_indices[g]);
    }
  } else {
    for (size_t g = 0; g < group_count; ++g) {
      std::vector<uint32_t> group_indices;
      group_indices.reserve(streamlines / group_count + 1);
      for (size_t i = g; i < streamlines; i += group_count) {
        group_indices.push_back(static_cast<uint32_t>(i));
      }
      trx.add_group_from_indices(group_names[g], group_indices);
    }
  }
}

std::unique_ptr<trx::TrxFile<half>> build_prefix_subset_trx(size_t streamlines,
                                                            GroupScenario scenario,
                                                            bool add_dps,
                                                            bool add_dpv) {
  if (g_reference_trx_path.empty()) {
    throw std::runtime_error("Reference TRX path not set.");
  }
  auto ref_trx = trx::load<half>(g_reference_trx_path);
  const size_t ref_count = ref_trx->num_streamlines();
  if (streamlines > ref_count) {
    throw std::runtime_error("Requested " + std::to_string(streamlines) +
                             " streamlines but reference only has " + std::to_string(ref_count));
  }

  const auto ids = build_prefix_ids(streamlines);
  auto out = ref_trx->subset_streamlines(ids, false);

  // Benchmark scenario owns grouping; drop any inherited groups first.
  out->groups.clear();
  if (!out->_uncompressed_folder_handle.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(out->_uncompressed_folder_handle) / "groups", ec);
  }

  if (!add_dps) {
    out->data_per_streamline.clear();
    if (!out->_uncompressed_folder_handle.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(std::filesystem::path(out->_uncompressed_folder_handle) / "dps", ec);
    }
  } else if (out->data_per_streamline.empty()) {
    std::vector<float> ones(streamlines, 1.0f);
    out->add_dps_from_vector("sift_weights", "float32", ones);
  }

  if (add_dpv) {
    std::vector<float> dpv(out->num_vertices(), 0.5f);
    out->add_dpv_from_vector("dpv_random", "float32", dpv);
  } else {
    out->data_per_vertex.clear();
    if (!out->_uncompressed_folder_handle.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(std::filesystem::path(out->_uncompressed_folder_handle) / "dpv", ec);
    }
  }

  assign_groups_to_trx(*out, scenario, streamlines);

  return out;
}

struct TrxWriteStats {
  double write_ms = 0.0;
  double file_size_bytes = 0.0;
};

struct RssSample {
  double elapsed_ms = 0.0;
  double rss_kb = 0.0;
  std::string phase;
};

struct FileSizeScenario {
  size_t streamlines = 0;
  bool add_dps = false;
  bool add_dpv = false;
  zip_uint32_t compression = ZIP_CM_STORE;
};

std::mutex g_rss_samples_mutex;

void append_rss_samples(const FileSizeScenario &scenario, const std::vector<RssSample> &samples) {
  if (samples.empty()) {
    return;
  }
  const char *path = std::getenv("TRX_RSS_SAMPLES_PATH");
  if (!path || path[0] == '\0') {
    return;
  }
  std::lock_guard<std::mutex> lock(g_rss_samples_mutex);
  std::ofstream out(path, std::ios::app);
  if (!out.is_open()) {
    return;
  }

  out << "{"
      << "\"streamlines\":" << scenario.streamlines << ","
      << "\"dps\":" << (scenario.add_dps ? 1 : 0) << ","
      << "\"dpv\":" << (scenario.add_dpv ? 1 : 0) << ","
      << "\"compression\":" << (scenario.compression == ZIP_CM_DEFLATE ? 1 : 0) << ","
      << "\"samples\":[";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "{"
        << "\"elapsed_ms\":" << samples[i].elapsed_ms << ","
        << "\"rss_kb\":" << samples[i].rss_kb << ","
        << "\"phase\":\"" << samples[i].phase << "\""
        << "}";
  }
  out << "]}\n";
}

std::mutex g_cleanup_mutex;
std::vector<std::string> g_cleanup_paths;
pid_t g_cleanup_owner_pid = 0;
bool g_cleanup_only_on_success = true;
bool g_run_success = false;

void cleanup_temp_paths() {
  if (g_cleanup_only_on_success && !g_run_success) {
    return;
  }
  if (g_cleanup_owner_pid != 0 && getpid() != g_cleanup_owner_pid) {
    return;
  }
  std::error_code ec;
  for (const auto &p : g_cleanup_paths) {
    std::filesystem::remove_all(p, ec);
  }
}

void register_cleanup(const std::string &path) {
  static bool registered = false;
  {
    std::lock_guard<std::mutex> lock(g_cleanup_mutex);
    if (g_cleanup_owner_pid == 0) {
      g_cleanup_owner_pid = getpid();
    }
    g_cleanup_paths.push_back(path);
  }
  if (!registered) {
    registered = true;
    std::atexit(cleanup_temp_paths);
  }
}

TrxWriteStats run_trx_file_size(size_t streamlines,
                                bool add_dps,
                                bool add_dpv,
                                zip_uint32_t compression) {
  const size_t progress_every = parse_env_size("TRX_BENCH_LOG_PROGRESS_EVERY", 0);

  const bool collect_rss = std::getenv("TRX_RSS_SAMPLES_PATH") != nullptr;
  const size_t sample_every = parse_env_size("TRX_RSS_SAMPLE_EVERY", 50000);
  const int sample_interval_ms = parse_env_int("TRX_RSS_SAMPLE_MS", 500);
  std::vector<RssSample> samples;
  std::mutex samples_mutex;
  const auto bench_start = std::chrono::steady_clock::now();
  auto record_sample = [&](const std::string &phase) {
    if (!collect_rss) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = now - bench_start;
    std::lock_guard<std::mutex> lock(samples_mutex);
    samples.push_back({elapsed.count(), get_max_rss_kb(), phase});
  };

  auto trx_subset = build_prefix_subset_trx(streamlines, GroupScenario::None, add_dps, add_dpv);
  if (progress_every > 0) {
    std::cerr << "[trx-bench] progress file_size streamlines=" << streamlines << " / " << streamlines << std::endl;
  }
  if (collect_rss && sample_every > 0) {
    record_sample("generate");
  }

  const std::string out_path = make_temp_path("trx_size");
  record_sample("before_finalize");

  std::atomic<bool> sampling{false};
  std::thread sampler;
  if (collect_rss) {
    sampling.store(true, std::memory_order_relaxed);
    sampler = std::thread([&]() {
      while (sampling.load(std::memory_order_relaxed)) {
        record_sample("finalize");
        std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms));
      }
    });
  }

  trx::TrxSaveOptions save_opts;
  save_opts.compression_standard = compression;
  const auto start = std::chrono::steady_clock::now();
  trx_subset->save(out_path, save_opts);
  const auto end = std::chrono::steady_clock::now();
  trx_subset->close();

  if (collect_rss) {
    sampling.store(false, std::memory_order_relaxed);
    if (sampler.joinable()) {
      sampler.join();
    }
  }
  record_sample("after_finalize");

  TrxWriteStats stats;
  stats.write_ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::error_code size_ec;
  const auto size = std::filesystem::file_size(out_path, size_ec);
  stats.file_size_bytes = size_ec ? 0.0 : static_cast<double>(size);
  std::error_code ec;
  std::filesystem::remove(out_path, ec);

  if (collect_rss) {
    FileSizeScenario scenario;
    scenario.streamlines = streamlines;
    scenario.add_dps = add_dps;
    scenario.add_dpv = add_dpv;
    scenario.compression = compression;
    append_rss_samples(scenario, samples);
  }
  return stats;
}

struct TrxOnDisk {
  std::string path;
  size_t streamlines = 0;
  size_t vertices = 0;
  double shard_merge_ms = 0.0;
  size_t shard_processes = 1;
};

// Parse the per-element column count and byte size encoded in a TRX filename.
// TRX convention: <name>.<ncols>.<dtype> or <name>.<dtype>
// e.g. "positions.3.float16" -> {3, 2}, "sift_weights.float32" -> {1, 4}
static std::pair<size_t, size_t> parse_trx_array_dims(const std::string &filename) {
  std::vector<std::string> parts;
  std::istringstream ss(filename);
  std::string tok;
  while (std::getline(ss, tok, '.')) {
    if (!tok.empty()) parts.push_back(tok);
  }
  if (parts.size() < 2) return {1, 4};
  const std::string dtype_str = parts.back();
  const size_t elem_size = static_cast<size_t>(trx::detail::_sizeof_dtype(dtype_str));
  if (parts.size() >= 3) {
    const std::string &maybe_ncols = parts[parts.size() - 2];
    if (!maybe_ncols.empty() &&
        std::all_of(maybe_ncols.begin(), maybe_ncols.end(), [](unsigned char c) { return std::isdigit(c); })) {
      return {static_cast<size_t>(std::stoul(maybe_ncols)), elem_size};
    }
  }
  return {1, elem_size};
}

#if defined(__unix__) || defined(__APPLE__)
static void truncate_file_to(const std::string &path, off_t byte_size) {
  if (::truncate(path.c_str(), byte_size) != 0) {
    throw std::runtime_error("truncate " + path + ": " + std::strerror(errno));
  }
}
#endif

// Truncate every regular file in dir to row_count rows based on the per-file dtype/ncols.
static void truncate_array_dir(const std::string &dir_path, size_t row_count) {
  std::error_code ec;
  if (!trx::fs::exists(dir_path, ec)) return;
  for (const auto &entry : trx::fs::directory_iterator(dir_path, ec)) {
    if (ec || !entry.is_regular_file()) continue;
    const auto [ncols, elem_size] = parse_trx_array_dims(entry.path().filename().string());
    truncate_file_to(entry.path().string(), static_cast<off_t>(row_count * ncols * elem_size));
  }
}

TrxOnDisk build_trx_file_on_disk_single(size_t streamlines,
                                        GroupScenario scenario,
                                        bool add_dps,
                                        bool add_dpv,
                                        zip_uint32_t compression,
                                        const std::string &out_path_override = "") {
  const size_t progress_every = parse_env_size("TRX_BENCH_LOG_PROGRESS_EVERY", 0);

  if (g_reference_trx_path.empty()) {
    throw std::runtime_error("Reference TRX path not set.");
  }
  const bool is_full_reference = (streamlines == g_reference_streamline_count);

  // Fast path: load the reference (extracts to a fresh unique temp dir each call), optionally
  // ftruncate the positions/offsets/dps files to a prefix boundary in O(1), then assign
  // groups and save. Avoids the large intermediate positions write from subset_streamlines().
  //
  // Used whenever dpv isn't needed, or when the full reference already has dpv (preserved
  // through the copy without any vector allocation).
  // Falls back to build_prefix_subset_trx only for synthetic-dpv at small sizes.
  if (!add_dpv || (is_full_reference && g_reference_has_dpv)) {
    log_bench_start("build_trx_prefix_fast", "streamlines=" + std::to_string(streamlines));

    // Load reference into a fresh temp dir; each call gets its own unique extraction.
    auto ref = trx::load<half>(g_reference_trx_path);
    const std::string temp_dir = ref->_uncompressed_folder_handle;

    // Locate positions and offsets files while the mmap is still open.
    std::string pos_path, off_path;
    {
      std::error_code ec;
      for (const auto &entry : trx::fs::directory_iterator(temp_dir, ec)) {
        if (ec) break;
        const std::string fn = entry.path().filename().string();
        if (fn.rfind("positions", 0) == 0) pos_path = entry.path().string();
        else if (fn.rfind("offsets", 0) == 0) off_path = entry.path().string();
      }
    }
    if (pos_path.empty() || off_path.empty()) {
      throw std::runtime_error("positions/offsets not found in " + temp_dir);
    }

    // Read vertex_cutoff directly from the offsets file so the dtype (uint32 vs uint64)
    // is always respected, regardless of how the Eigen map interprets the mmap width.
    const auto [off_ncols, off_elem] =
        parse_trx_array_dims(trx::fs::path(off_path).filename().string());
    size_t vertex_cutoff = 0;
    {
      std::ifstream ofs(off_path, std::ios::binary);
      ofs.seekg(static_cast<std::streamoff>(streamlines * off_ncols * off_elem));
      if (off_elem == 4) {
        uint32_t v = 0;
        ofs.read(reinterpret_cast<char *>(&v), 4);
        vertex_cutoff = static_cast<size_t>(v);
      } else {
        uint64_t v = 0;
        ofs.read(reinterpret_cast<char *>(&v), 8);
        vertex_cutoff = static_cast<size_t>(v);
      }
    }

    // Release mmaps without deleting temp_dir — we own it from here.
    ref->_owns_uncompressed_folder = false;
    ref.reset();

    if (!is_full_reference) {
      // Truncate positions and offsets to the prefix boundary.
      const auto [pos_ncols, pos_elem] =
          parse_trx_array_dims(trx::fs::path(pos_path).filename().string());
      truncate_file_to(pos_path, static_cast<off_t>(vertex_cutoff * pos_ncols * pos_elem));
      truncate_file_to(off_path, static_cast<off_t>((streamlines + 1) * off_ncols * off_elem));

      // Truncate DPS arrays to the prefix streamline count.
      truncate_array_dir(temp_dir + trx::SEPARATOR + "dps", streamlines);

      // DPV is absent at this point (condition above guarantees !add_dpv for non-full ref).
      std::error_code ec2;
      std::filesystem::remove_all(trx::fs::path(temp_dir) / "dpv", ec2);

      // Patch NB_STREAMLINES and NB_VERTICES in header.json.
      const std::string header_path = temp_dir + trx::SEPARATOR + "header.json";
      {
        std::ifstream in(header_path);
        std::string raw((std::istreambuf_iterator<char>(in)), {});
        std::string parse_err;
        json hdr = json::parse(raw, parse_err);
        if (!parse_err.empty()) throw std::runtime_error("header.json parse error: " + parse_err);
        hdr = trx::_json_set(hdr, "NB_STREAMLINES", static_cast<int>(streamlines));
        hdr = trx::_json_set(hdr, "NB_VERTICES", static_cast<int>(vertex_cutoff));
        std::ofstream out(header_path, std::ios::trunc);
        out << hdr.dump();
      }
    }

    // Strip DPS if not wanted (applies to both full and prefix).
    if (!add_dps) {
      std::error_code ec;
      std::filesystem::remove_all(trx::fs::path(temp_dir) / "dps", ec);
    }

    // Clear any groups the reference may carry; the benchmark owns grouping.
    {
      std::error_code ec;
      std::filesystem::remove_all(trx::fs::path(temp_dir) / "groups", ec);
    }

    auto trx = trx::TrxFile<half>::load_from_directory(temp_dir);

    // Add synthetic DPS if requested but the reference didn't have it.
    if (add_dps && trx->data_per_streamline.empty()) {
      std::vector<float> ones(streamlines, 1.0f);
      trx->add_dps_from_vector("sift_weights", "float32", ones);
    }

    assign_groups_to_trx(*trx, scenario, streamlines);

    const std::string out_path =
        out_path_override.empty() ? make_temp_path("trx_input") : out_path_override;
    trx::TrxSaveOptions save_opts;
    save_opts.compression_standard = compression;
    trx->save(out_path, save_opts);
    const size_t total_vertices = trx->num_vertices();
    trx->close();
    trx::rm_dir(temp_dir);

    if (out_path_override.empty()) {
      register_cleanup(out_path);
    }

    log_bench_end("build_trx_prefix_fast", "streamlines=" + std::to_string(streamlines));
    return {out_path, streamlines, total_vertices, 0.0, 1};
  }

  // Fallback: prefix-subset path for synthetic dpv at smaller streamline counts.
  auto trx_subset = build_prefix_subset_trx(streamlines, scenario, add_dps, add_dpv);
  const size_t total_vertices = trx_subset->num_vertices();
  const std::string out_path =
      out_path_override.empty() ? make_temp_path("trx_input") : out_path_override;
  trx::TrxSaveOptions save_opts;
  save_opts.compression_standard = compression;
  trx_subset->save(out_path, save_opts);
  trx_subset->close();
  if (progress_every > 0 &&
      (parse_env_bool("TRX_BENCH_CHILD_LOG", false) || parse_env_bool("TRX_BENCH_LOG", false))) {
    std::cerr << "[trx-bench] progress build_trx streamlines=" << streamlines << " / "
              << streamlines << std::endl;
  }
  if (out_path_override.empty()) {
    register_cleanup(out_path);
  }
  return {out_path, streamlines, total_vertices, 0.0, 1};
}

TrxOnDisk build_trx_file_on_disk(size_t streamlines,
                                 GroupScenario scenario,
                                 bool add_dps,
                                 bool add_dpv,
                                 zip_uint32_t compression) {
  return build_trx_file_on_disk_single(streamlines, scenario, add_dps, add_dpv, compression);
}

struct QueryDataset {
  std::unique_ptr<trx::TrxFile<half>> trx;
  std::vector<std::array<float, 3>> slab_mins;
  std::vector<std::array<float, 3>> slab_maxs;
};

void build_slabs(std::vector<std::array<float, 3>> &mins, std::vector<std::array<float, 3>> &maxs) {
  mins.clear();
  maxs.clear();
  mins.reserve(kSlabCount);
  maxs.reserve(kSlabCount);
  const float z_range = kFov.max_z - kFov.min_z;
  for (size_t i = 0; i < kSlabCount; ++i) {
    const float t = (kSlabCount == 1) ? 0.5f : static_cast<float>(i) / static_cast<float>(kSlabCount - 1);
    const float center_z = kFov.min_z + t * z_range;
    const float min_z = std::max(kFov.min_z, center_z - kSlabThicknessMm * 0.5f);
    const float max_z = std::min(kFov.max_z, center_z + kSlabThicknessMm * 0.5f);
    mins.push_back({kFov.min_x, kFov.min_y, min_z});
    maxs.push_back({kFov.max_x, kFov.max_y, max_z});
  }
}

struct ScenarioParams {
  size_t streamlines = 0;
  GroupScenario scenario = GroupScenario::None;
  bool add_dps = false;
  bool add_dpv = false;
};

struct KeyHash {
  using Key = std::tuple<size_t, int, int, int>;
  size_t operator()(const Key &key) const {
    size_t h = 0;
    auto hash_combine = [&](size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    hash_combine(std::hash<size_t>{}(std::get<0>(key)));
    hash_combine(std::hash<int>{}(std::get<1>(key)));
    hash_combine(std::hash<int>{}(std::get<2>(key)));
    hash_combine(std::hash<int>{}(std::get<3>(key)));
    return h;
  }
};

void maybe_write_query_timings(const ScenarioParams &scenario, const std::vector<double> &timings_ms) {
  static std::mutex mutex;
  static std::unordered_set<KeyHash::Key, KeyHash> seen;
  const KeyHash::Key key{scenario.streamlines,
                         static_cast<int>(scenario.scenario),
                         scenario.add_dps ? 1 : 0,
                         scenario.add_dpv ? 1 : 0};

  std::lock_guard<std::mutex> lock(mutex);
  if (!seen.insert(key).second) {
    return;
  }

  const char *env_path = std::getenv("TRX_QUERY_TIMINGS_PATH");
  const std::filesystem::path out_path = env_path ? env_path : "bench/query_timings.jsonl";
  std::error_code ec;
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path(), ec);
  }
  std::ofstream out(out_path, std::ios::app);
  if (!out.is_open()) {
    return;
  }

  out << "{"
      << "\"streamlines\":" << scenario.streamlines << ","
      << "\"group_case\":" << static_cast<int>(scenario.scenario) << ","
      << "\"group_count\":" << group_count_for(scenario.scenario) << ","
      << "\"dps\":" << (scenario.add_dps ? 1 : 0) << ","
      << "\"dpv\":" << (scenario.add_dpv ? 1 : 0) << ","
      << "\"slab_thickness_mm\":" << kSlabThicknessMm << ","
      << "\"timings_ms\":[";
  for (size_t i = 0; i < timings_ms.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << timings_ms[i];
  }
  out << "]}\n";
}
} // namespace

static void BM_TrxFileSize_Float16(benchmark::State &state) {
  const size_t streamlines = static_cast<size_t>(state.range(0));
  const auto scenario = static_cast<GroupScenario>(state.range(1));
  const bool add_dps = state.range(2) != 0;
  const bool add_dpv = state.range(3) != 0;
  const bool use_zip = state.range(4) != 0;
  const auto compression = use_zip ? ZIP_CM_DEFLATE : ZIP_CM_STORE;
  const size_t skip_zip_at = parse_env_size("TRX_BENCH_SKIP_ZIP_AT", 5000000);
  const size_t skip_dpv_at = parse_env_size("TRX_BENCH_SKIP_DPV_AT", 10000000);
  if (use_zip && streamlines >= skip_zip_at) {
    state.SkipWithMessage("zip compression skipped for large streamlines");
    return;
  }
  // Skip synthetic dpv at high streamline counts due to memory cost (~40-50 GB).
  // Exception: if the reference already has dpv and this is the full-reference count, the
  // fast path handles it via ftruncate without allocating a dpv vector.
  const bool fast_path_covers_dpv =
      (streamlines == g_reference_streamline_count && g_reference_has_dpv);
  if (add_dpv && streamlines >= skip_dpv_at && !fast_path_covers_dpv) {
    state.SkipWithMessage("dpv skipped: requires 40-50 GB memory (set TRX_BENCH_SKIP_DPV_AT=0 to force)");
    return;
  }
  log_bench_start("BM_TrxFileSize_Float16",
                  "streamlines=" + std::to_string(streamlines) +
                      " group_case=" + std::to_string(state.range(1)) +
                      " dps=" + std::to_string(static_cast<int>(add_dps)) +
                      " dpv=" + std::to_string(static_cast<int>(add_dpv)) +
                      " compression=" + std::to_string(static_cast<int>(use_zip)));

  double total_write_ms = 0.0;
  double total_file_bytes = 0.0;
  double total_merge_ms = 0.0;
  double total_build_ms = 0.0;
  double total_merge_processes = 0.0;
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    const auto on_disk =
        build_trx_file_on_disk(streamlines, scenario, add_dps, add_dpv, compression);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    total_build_ms += elapsed.count();
    total_merge_ms += on_disk.shard_merge_ms;
    total_merge_processes += static_cast<double>(on_disk.shard_processes);
    total_write_ms += elapsed.count();
    total_file_bytes += static_cast<double>(file_size_bytes(on_disk.path));
  }

  state.counters["streamlines"] = static_cast<double>(streamlines);
  state.counters["group_case"] = static_cast<double>(state.range(1));
  state.counters["group_count"] = static_cast<double>(group_count_for(scenario));
  state.counters["dps"] = add_dps ? 1.0 : 0.0;
  state.counters["dpv"] = add_dpv ? 1.0 : 0.0;
  state.counters["compression"] = use_zip ? 1.0 : 0.0;
  state.counters["positions_dtype"] = 16.0;
  state.counters["write_ms"] = total_write_ms / static_cast<double>(state.iterations());
  state.counters["build_ms"] = total_build_ms / static_cast<double>(state.iterations());
  if (total_merge_ms > 0.0) {
    state.counters["shard_merge_ms"] = total_merge_ms / static_cast<double>(state.iterations());
    state.counters["shard_processes"] = total_merge_processes / static_cast<double>(state.iterations());
  }
  state.counters["file_bytes"] = total_file_bytes / static_cast<double>(state.iterations());
  state.counters["max_rss_kb"] = get_max_rss_kb();

  log_bench_end("BM_TrxFileSize_Float16",
                "streamlines=" + std::to_string(streamlines));
}

static void BM_TrxStream_TranslateWrite(benchmark::State &state) {
  const size_t streamlines = static_cast<size_t>(state.range(0));
  const auto scenario = static_cast<GroupScenario>(state.range(1));
  const bool add_dps = state.range(2) != 0;
  const bool add_dpv = state.range(3) != 0;
  const size_t progress_every = parse_env_size("TRX_BENCH_LOG_PROGRESS_EVERY", 0);
  log_bench_start("BM_TrxStream_TranslateWrite",
                  "streamlines=" + std::to_string(streamlines) + " group_case=" + std::to_string(state.range(1)) +
                      " dps=" + std::to_string(static_cast<int>(add_dps)) +
                      " dpv=" + std::to_string(static_cast<int>(add_dpv)));

  using Key = KeyHash::Key;
  static std::unordered_map<Key, TrxOnDisk, KeyHash> cache;

  const Key key{streamlines, static_cast<int>(scenario), add_dps ? 1 : 0, add_dpv ? 1 : 0};
  if (cache.find(key) == cache.end()) {
    state.PauseTiming();
    cache.emplace(key,
                  build_trx_file_on_disk(streamlines, scenario, add_dps, add_dpv, ZIP_CM_STORE));
    state.ResumeTiming();
  }

  const auto &dataset = cache.at(key);
  if (dataset.shard_processes > 1 && dataset.shard_merge_ms > 0.0) {
    state.counters["shard_merge_ms"] = dataset.shard_merge_ms;
    state.counters["shard_processes"] = static_cast<double>(dataset.shard_processes);
  }
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    auto trx = trx::load_any(dataset.path);
    const size_t chunk_bytes = parse_env_size("TRX_BENCH_CHUNK_BYTES", 1024ULL * 1024ULL * 1024ULL);
    const std::string out_dir = make_work_dir_name("trx_translate_chunk");
    trx::PrepareOutputOptions prep_opts;
    prep_opts.overwrite_existing = true;
    const auto out_info = trx::prepare_positions_output(trx, out_dir, prep_opts);

    std::ofstream out_positions(out_info.positions_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out_positions.is_open()) {
      throw std::runtime_error("Failed to open output positions file: " + out_info.positions_path);
    }

    trx.for_each_positions_chunk(chunk_bytes,
                                 [&](trx::TrxScalarType dtype, const void *data, size_t offset, size_t count) {
                                   (void)offset;
                                   if (progress_every > 0 && ((offset + count) % progress_every == 0)) {
                                     std::cerr << "[trx-bench] progress translate points=" << (offset + count)
                                               << " / " << out_info.points << std::endl;
                                   }
                                   const size_t total_vals = count * 3;
                                   if (dtype == trx::TrxScalarType::Float16) {
                                     const auto *src = reinterpret_cast<const Eigen::half *>(data);
                                     std::vector<Eigen::half> tmp(total_vals);
                                     for (size_t i = 0; i < total_vals; ++i) {
                                       tmp[i] = static_cast<Eigen::half>(static_cast<float>(src[i]) + 1.0f);
                                     }
                                     out_positions.write(reinterpret_cast<const char *>(tmp.data()),
                                                         static_cast<std::streamsize>(tmp.size() * sizeof(Eigen::half)));
                                   } else if (dtype == trx::TrxScalarType::Float32) {
                                     const auto *src = reinterpret_cast<const float *>(data);
                                     std::vector<float> tmp(total_vals);
                                     for (size_t i = 0; i < total_vals; ++i) {
                                       tmp[i] = src[i] + 1.0f;
                                     }
                                     out_positions.write(reinterpret_cast<const char *>(tmp.data()),
                                                         static_cast<std::streamsize>(tmp.size() * sizeof(float)));
                                   } else {
                                     const auto *src = reinterpret_cast<const double *>(data);
                                     std::vector<double> tmp(total_vals);
                                     for (size_t i = 0; i < total_vals; ++i) {
                                       tmp[i] = src[i] + 1.0;
                                     }
                                     out_positions.write(reinterpret_cast<const char *>(tmp.data()),
                                                         static_cast<std::streamsize>(tmp.size() * sizeof(double)));
                                   }
                                 });
    out_positions.flush();
    out_positions.close();

    const std::string out_path = make_temp_path("trx_translate");
    trx::TrxSaveOptions save_opts;
    save_opts.mode = trx::TrxSaveMode::Archive;
    save_opts.compression_standard = ZIP_CM_STORE;
    save_opts.overwrite_existing = true;
    trx.save(out_path, save_opts);
    trx::rm_dir(out_dir);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    state.SetIterationTime(elapsed.count());

    std::error_code ec;
    std::filesystem::remove(out_path, ec);
    benchmark::DoNotOptimize(trx);
  }

  state.counters["streamlines"] = static_cast<double>(streamlines);
  state.counters["group_case"] = static_cast<double>(state.range(1));
  state.counters["group_count"] = static_cast<double>(group_count_for(scenario));
  state.counters["dps"] = add_dps ? 1.0 : 0.0;
  state.counters["dpv"] = add_dpv ? 1.0 : 0.0;
  state.counters["positions_dtype"] = 16.0;
  state.counters["max_rss_kb"] = get_max_rss_kb();

  log_bench_end("BM_TrxStream_TranslateWrite",
                "streamlines=" + std::to_string(streamlines) + " group_case=" + std::to_string(state.range(1)));
}

static void BM_TrxQueryAabb_Slabs(benchmark::State &state) {
  const size_t streamlines = static_cast<size_t>(state.range(0));
  const auto scenario = static_cast<GroupScenario>(state.range(1));
  const bool add_dps = state.range(2) != 0;
  const bool add_dpv = state.range(3) != 0;
  log_bench_start("BM_TrxQueryAabb_Slabs",
                  "streamlines=" + std::to_string(streamlines) + " group_case=" + std::to_string(state.range(1)) +
                      " dps=" + std::to_string(static_cast<int>(add_dps)) +
                      " dpv=" + std::to_string(static_cast<int>(add_dpv)));

  using Key = KeyHash::Key;
  static std::unordered_map<Key, QueryDataset, KeyHash> cache;

  const Key key{streamlines, static_cast<int>(scenario), add_dps ? 1 : 0, add_dpv ? 1 : 0};
  if (cache.find(key) == cache.end()) {
    state.PauseTiming();
    QueryDataset dataset;
    auto on_disk = build_trx_file_on_disk(streamlines, scenario, add_dps, add_dpv, ZIP_CM_STORE);
    dataset.trx = trx::load<half>(on_disk.path);
    dataset.trx->get_or_build_streamline_aabbs();
    build_slabs(dataset.slab_mins, dataset.slab_maxs);
    cache.emplace(key, std::move(dataset));
    state.ResumeTiming();
  }

  auto &dataset = cache.at(key);
  for (auto _ : state) {
    std::vector<double> slab_times_ms;
    slab_times_ms.reserve(kSlabCount);

    const auto start = std::chrono::steady_clock::now();
    size_t total = 0;
    for (size_t i = 0; i < kSlabCount; ++i) {
      const auto &min_corner = dataset.slab_mins[i];
      const auto &max_corner = dataset.slab_maxs[i];
      const auto q_start = std::chrono::steady_clock::now();
      auto subset = dataset.trx->query_aabb(min_corner, max_corner);
      const auto q_end = std::chrono::steady_clock::now();
      const std::chrono::duration<double, std::milli> q_elapsed = q_end - q_start;
      slab_times_ms.push_back(q_elapsed.count());
      total += subset->num_streamlines();
      subset->close();
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    state.SetIterationTime(elapsed.count());
    benchmark::DoNotOptimize(total);

    auto sorted = slab_times_ms;
    std::sort(sorted.begin(), sorted.end());
    const auto p50 = sorted[sorted.size() / 2];
    const auto p95_idx = static_cast<size_t>(std::ceil(0.95 * sorted.size())) - 1;
    const auto p95 = sorted[std::min(p95_idx, sorted.size() - 1)];
    state.counters["query_p50_ms"] = p50;
    state.counters["query_p95_ms"] = p95;

    ScenarioParams params;
    params.streamlines = streamlines;
    params.scenario = scenario;
    params.add_dps = add_dps;
    params.add_dpv = add_dpv;
    maybe_write_query_timings(params, slab_times_ms);
  }

  state.counters["streamlines"] = static_cast<double>(streamlines);
  state.counters["group_case"] = static_cast<double>(state.range(1));
  state.counters["group_count"] = static_cast<double>(group_count_for(scenario));
  state.counters["dps"] = add_dps ? 1.0 : 0.0;
  state.counters["dpv"] = add_dpv ? 1.0 : 0.0;
  state.counters["query_count"] = static_cast<double>(kSlabCount);
  state.counters["slab_thickness_mm"] = kSlabThicknessMm;
  state.counters["positions_dtype"] = 16.0;
  state.counters["max_rss_kb"] = get_max_rss_kb();

  log_bench_end("BM_TrxQueryAabb_Slabs",
                "streamlines=" + std::to_string(streamlines) + " group_case=" + std::to_string(state.range(1)));
}

static void ApplySizeArgs(benchmark::internal::Benchmark *bench) {
  const std::array<int, 2> flags = {0, 1};
  const bool core_profile = is_core_profile();
  const size_t dpv_limit = core_dpv_max_streamlines();
  const size_t zip_limit = core_zip_max_streamlines();
  const auto counts_desc = streamlines_for_benchmarks();
  const auto groups = group_cases_for_benchmarks();
  for (const auto count : counts_desc) {
    const std::vector<int> dpv_flags = (!core_profile || count <= dpv_limit)
                                           ? std::vector<int>{0, 1}
                                           : std::vector<int>{0};
    const std::vector<int> compression_flags = (!core_profile || count <= zip_limit)
                                                   ? std::vector<int>{0, 1}
                                                   : std::vector<int>{0};
    for (const auto group_case : groups) {
      for (const auto dps : flags) {
        for (const auto dpv : dpv_flags) {
          for (const auto compression : compression_flags) {
            bench->Args({static_cast<int64_t>(count), group_case, dps, dpv, compression});
          }
        }
      }
    }
  }
}

static void ApplyStreamArgs(benchmark::internal::Benchmark *bench) {
  const std::array<int, 2> flags = {0, 1};
  const bool core_profile = is_core_profile();
  const size_t dpv_limit = core_dpv_max_streamlines();
  const auto groups = group_cases_for_benchmarks();
  const auto counts_desc = streamlines_for_benchmarks();
  for (const auto count : counts_desc) {
    const std::vector<int> dpv_flags = (!core_profile || count <= dpv_limit)
                                           ? std::vector<int>{0, 1}
                                           : std::vector<int>{0};
    for (const auto group_case : groups) {
      for (const auto dps : flags) {
        for (const auto dpv : dpv_flags) {
          bench->Args({static_cast<int64_t>(count), group_case, dps, dpv});
        }
      }
    }
  }
}

static void ApplyQueryArgs(benchmark::internal::Benchmark *bench) {
  const std::array<int, 2> flags = {0, 1};
  const bool core_profile = is_core_profile();
  const size_t dpv_limit = core_dpv_max_streamlines();
  const auto groups = group_cases_for_benchmarks();
  const auto counts_desc = streamlines_for_benchmarks();
  for (const auto count : counts_desc) {
    const std::vector<int> dpv_flags = (!core_profile || count <= dpv_limit)
                                           ? std::vector<int>{0, 1}
                                           : std::vector<int>{0};
    for (const auto group_case : groups) {
      for (const auto dps : flags) {
        for (const auto dpv : dpv_flags) {
          bench->Args({static_cast<int64_t>(count), group_case, dps, dpv});
        }
      }
    }
  }
  bench->Iterations(1);
}

BENCHMARK(BM_TrxFileSize_Float16)
    ->Apply(ApplySizeArgs)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_TrxStream_TranslateWrite)
    ->Apply(ApplyStreamArgs)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_TrxQueryAabb_Slabs)
    ->Apply(ApplyQueryArgs)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

int main(int argc, char **argv) {
  // Parse custom flags before benchmark::Initialize
  bool verbose = false;
  bool show_help = false;
  std::string reference_trx;
  
  // First pass: detect custom flags
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--help-custom") {
      show_help = true;
    } else if (arg == "--reference-trx" && i + 1 < argc) {
      reference_trx = argv[i + 1];
      ++i;  // Skip next arg since it's the value
    }
  }
  
  if (show_help) {
    std::cout << "\nCustom benchmark options:\n"
              << "  --reference-trx PATH   Path to reference TRX file for sampling (REQUIRED)\n"
              << "  --verbose, -v          Enable verbose progress logging (prints every 50k streamlines)\n"
              << "                         Equivalent to: TRX_BENCH_LOG=1 TRX_BENCH_CHILD_LOG=1 \n"
              << "                         TRX_BENCH_LOG_PROGRESS_EVERY=50000\n"
              << "  --help-custom          Show this help message\n"
              << "\nFor standard benchmark options, use --help\n"
              << std::endl;
    return 0;
  }
  
  // Validate reference TRX path
  if (reference_trx.empty()) {
    std::cerr << "Error: --reference-trx flag is required\n"
              << "Usage: " << argv[0] << " --reference-trx <path_to_trx_file> [benchmark_options]\n"
              << "Use --help-custom for more information\n" << std::endl;
    return 1;
  }
  
  // Check if reference file exists
  std::error_code ec;
  if (!std::filesystem::exists(reference_trx, ec)) {
    std::cerr << "Error: Reference TRX file not found: " << reference_trx << std::endl;
    return 1;
  }
  
  // Set global reference path
  g_reference_trx_path = reference_trx;
  std::cerr << "[trx-bench] Using reference TRX: " << g_reference_trx_path << std::endl;

  // Pre-inspect reference to avoid repeated loads and to inform skip decisions
  {
    auto ref = trx::load<half>(g_reference_trx_path);
    g_reference_streamline_count = ref->num_streamlines();
    g_reference_has_dpv = !ref->data_per_vertex.empty();
    ref->close();
    std::cerr << "[trx-bench] Reference: " << g_reference_streamline_count
              << " streamlines, dpv=" << (g_reference_has_dpv ? "yes" : "no") << std::endl;
  }
  
  // Enable verbose logging if requested
  if (verbose) {
    setenv("TRX_BENCH_LOG", "1", 0);  // Don't override if already set
    setenv("TRX_BENCH_CHILD_LOG", "1", 0);
    if (std::getenv("TRX_BENCH_LOG_PROGRESS_EVERY") == nullptr) {
      setenv("TRX_BENCH_LOG_PROGRESS_EVERY", "50000", 1);
    }
    std::cerr << "[trx-bench] Verbose mode enabled (progress every "
              << parse_env_size("TRX_BENCH_LOG_PROGRESS_EVERY", 50000) 
              << " streamlines)\n" << std::endl;
  }
  
  // Second pass: remove custom flags from argv before passing to benchmark::Initialize
  std::vector<char*> filtered_argv;
  filtered_argv.push_back(argv[0]);  // Keep program name
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--verbose" || arg == "-v" || arg == "--help-custom") {
      continue;
    } else if (arg == "--reference-trx") {
      ++i;  // Skip the next arg (the path value)
      continue;
    }
    filtered_argv.push_back(argv[i]);
  }
  int filtered_argc = static_cast<int>(filtered_argv.size());
  
  ::benchmark::Initialize(&filtered_argc, filtered_argv.data());
  if (::benchmark::ReportUnrecognizedArguments(filtered_argc, filtered_argv.data())) {
    return 1;
  }
  try {
    ::benchmark::RunSpecifiedBenchmarks();
    g_run_success = true;
  } catch (const std::exception &ex) {
    std::cerr << "Benchmark failed: " << ex.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Benchmark failed with unknown exception." << std::endl;
    return 1;
  }
  return 0;
}
