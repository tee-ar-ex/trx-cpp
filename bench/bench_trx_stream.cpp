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

#include <zip.h>

namespace {
using Eigen::half;

constexpr float kMinLengthMm = 20.0f;
constexpr float kMaxLengthMm = 500.0f;
constexpr float kStepMm = 2.0f;
constexpr float kCurvatureSigma = 0.08f;
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
constexpr float kRandomMinMm = 10.0f;
constexpr float kRandomMaxMm = 400.0f;

enum class GroupScenario : int { None = 0, Bundles = 1, Connectome = 2 };
enum class LengthProfile : int { Mixed = 0, Short = 1, Medium = 2, Long = 3 };

constexpr size_t kBundleCount = 80;
constexpr size_t kConnectomeRegions = 100;

std::string make_temp_path(const std::string &prefix) {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
  const auto dir = std::filesystem::temp_directory_path();
  return (dir / (prefix + "_" + std::to_string(id) + ".trx")).string();
}

std::string make_temp_dir_name(const std::string &prefix) {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
  const auto dir = std::filesystem::temp_directory_path();
  return (dir / (prefix + "_" + std::to_string(id))).string();
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

std::string make_status_path(const std::string &prefix) {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
  const auto dir = std::filesystem::temp_directory_path();
  return (dir / (prefix + "_" + std::to_string(id) + ".txt")).string();
}

std::string make_temp_dir_path(const std::string &prefix) {
  return trx::make_temp_dir(prefix);
}

void register_cleanup(const std::string &path);
std::vector<std::string> list_files(const std::string &dir);

std::string find_file_by_prefix(const std::string &dir, const std::string &prefix) {
  std::error_code ec;
  for (const auto &entry : trx::fs::directory_iterator(dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (filename.rfind(prefix, 0) == 0) {
      return entry.path().string();
    }
  }
  return "";
}

std::vector<std::string> list_files(const std::string &dir) {
  std::vector<std::string> files;
  std::error_code ec;
  if (!trx::fs::exists(dir, ec)) {
    return files;
  }
  for (const auto &entry : trx::fs::directory_iterator(dir, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    files.push_back(entry.path().filename().string());
  }
  std::sort(files.begin(), files.end());
  return files;
}

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

void wait_for_shard_ok(const std::vector<std::string> &shard_paths,
                       const std::vector<std::string> &status_paths,
                       size_t timeout_ms) {
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    bool all_ok = true;
    for (size_t i = 0; i < shard_paths.size(); ++i) {
      const auto ok_path = trx::fs::path(shard_paths[i]) / "SHARD_OK";
      std::error_code ec;
      if (!trx::fs::exists(ok_path, ec)) {
        all_ok = false;
        break;
      }
    }
    if (all_ok) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    if (elapsed_ms > static_cast<long long>(timeout_ms)) {
      std::string detail = "Timed out waiting for SHARD_OK";
      for (size_t i = 0; i < status_paths.size(); ++i) {
        std::ifstream in(status_paths[i]);
        std::string line;
        if (in.is_open()) {
          std::getline(in, line);
        }
        if (!line.empty()) {
          detail += " shard_" + std::to_string(i) + "=" + line;
        }
      }
      throw std::runtime_error(detail);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void copy_file_append(const std::string &src, const std::string &dst, std::size_t buffer_bytes = 8 * 1024 * 1024) {
  std::ifstream in(src, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open file for read: " + src);
  }
  std::ofstream out(dst, std::ios::binary | std::ios::out | std::ios::app);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open file for append: " + dst);
  }
  std::vector<char> buffer(buffer_bytes);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = in.gcount();
    if (count > 0) {
      out.write(buffer.data(), count);
    }
  }
}

std::pair<size_t, size_t> read_header_counts(const std::string &dir) {
  const auto header_path = trx::fs::path(dir) / "header.json";
  std::ifstream in;
  for (int attempt = 0; attempt < 5; ++attempt) {
    in.open(header_path);
    if (in.is_open()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!in.is_open()) {
    std::error_code ec;
    const bool exists = trx::fs::exists(dir, ec);
    const auto files = list_files(dir);
    const int open_err = errno;
    std::string detail = "Failed to open header.json at: " + header_path.string();
    detail += " exists=" + std::string(exists ? "true" : "false");
    detail += " errno=" + std::to_string(open_err) + " msg=" + std::string(std::strerror(open_err));
    if (!files.empty()) {
      detail += " files=[";
      for (size_t i = 0; i < files.size(); ++i) {
        if (i > 0) {
          detail += ",";
        }
        detail += files[i];
      }
      detail += "]";
    }
    throw std::runtime_error(detail);
  }
  std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::string err;
  const auto header = json::parse(contents, err);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse header.json: " + err);
  }
  const auto nb_streamlines = static_cast<size_t>(header["NB_STREAMLINES"].int_value());
  const auto nb_vertices = static_cast<size_t>(header["NB_VERTICES"].int_value());
  return {nb_streamlines, nb_vertices};
}

json read_header_json(const std::string &dir) {
  const auto header_path = trx::fs::path(dir) / "header.json";
  std::ifstream in;
  for (int attempt = 0; attempt < 5; ++attempt) {
    in.open(header_path);
    if (in.is_open()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!in.is_open()) {
    std::error_code ec;
    const bool exists = trx::fs::exists(dir, ec);
    const auto files = list_files(dir);
    const int open_err = errno;
    std::string detail = "Failed to open header.json at: " + header_path.string();
    detail += " exists=" + std::string(exists ? "true" : "false");
    detail += " errno=" + std::to_string(open_err) + " msg=" + std::string(std::strerror(open_err));
    if (!files.empty()) {
      detail += " files=[";
      for (size_t i = 0; i < files.size(); ++i) {
        if (i > 0) {
          detail += ",";
        }
        detail += files[i];
      }
      detail += "]";
    }
    throw std::runtime_error(detail);
  }
  std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::string err;
  const auto header = json::parse(contents, err);
  if (!err.empty()) {
    throw std::runtime_error("Failed to parse header.json: " + err);
  }
  return header;
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

size_t group_count_for(GroupScenario scenario) {
  switch (scenario) {
  case GroupScenario::Bundles:
    return kBundleCount;
  case GroupScenario::Connectome:
    return (kConnectomeRegions * (kConnectomeRegions - 1)) / 2;
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

void log_bench_config(const std::string &name, size_t threads, size_t batch_size) {
  if (!parse_env_bool("TRX_BENCH_LOG", false)) {
    return;
  }
  std::cerr << "[trx-bench] config " << name << " threads=" << threads << " batch=" << batch_size << std::endl;
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
    names.reserve((kConnectomeRegions * (kConnectomeRegions - 1)) / 2);
    for (size_t i = 1; i <= kConnectomeRegions; ++i) {
      for (size_t j = i + 1; j <= kConnectomeRegions; ++j) {
        names.push_back("conn_" + std::to_string(i) + "_" + std::to_string(j));
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

float sample_length_mm(std::mt19937 &rng, LengthProfile profile) {
  auto sample_uniform = [&](float min_val, float max_val) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    return dist(rng);
  };
  switch (profile) {
  case LengthProfile::Short:
    return sample_uniform(20.0f, 120.0f);
  case LengthProfile::Medium:
    return sample_uniform(80.0f, 260.0f);
  case LengthProfile::Long:
    return sample_uniform(200.0f, 500.0f);
  case LengthProfile::Mixed:
  default:
    return sample_uniform(kMinLengthMm, kMaxLengthMm);
  }
}

size_t estimate_points_per_streamline(LengthProfile profile) {
  float mean_length = 0.0f;
  switch (profile) {
  case LengthProfile::Short:
    mean_length = 70.0f;
    break;
  case LengthProfile::Medium:
    mean_length = 170.0f;
    break;
  case LengthProfile::Long:
    mean_length = 350.0f;
    break;
  case LengthProfile::Mixed:
  default:
    mean_length = 260.0f;
    break;
  }
  return static_cast<size_t>(std::ceil(mean_length / kStepMm)) + 1;
}

std::array<float, 3> random_unit_vector(std::mt19937 &rng) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::array<float, 3> v{dist(rng), dist(rng), dist(rng)};
  const float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (norm < 1e-6f) {
    return {1.0f, 0.0f, 0.0f};
  }
  v[0] /= norm;
  v[1] /= norm;
  v[2] /= norm;
  return v;
}

std::vector<std::array<float, 3>> generate_streamline_points(std::mt19937 &rng, LengthProfile profile) {
  const float length_mm = sample_length_mm(rng, profile);
  const size_t point_count = std::max<size_t>(2, static_cast<size_t>(std::ceil(length_mm / kStepMm)) + 1);
  std::vector<std::array<float, 3>> points;
  points.reserve(point_count);

  std::uniform_real_distribution<float> dist_x(kRandomMinMm, kRandomMaxMm);
  std::uniform_real_distribution<float> dist_y(kRandomMinMm, kRandomMaxMm);
  std::uniform_real_distribution<float> dist_z(kRandomMinMm, kRandomMaxMm);

  for (size_t i = 0; i < point_count; ++i) {
    points.push_back({dist_x(rng), dist_y(rng), dist_z(rng)});
  }

  return points;
}

std::vector<std::array<float, 3>> generate_streamline_points_seeded(uint32_t seed, LengthProfile profile) {
  std::mt19937 rng(seed);
  return generate_streamline_points(rng, profile);
}

size_t bench_threads() {
  const size_t requested = parse_env_size("TRX_BENCH_THREADS", 0);
  if (requested > 0) {
    return requested;
  }
  const unsigned int hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1U : static_cast<size_t>(hc);
}

size_t bench_batch_size() {
  return parse_env_size("TRX_BENCH_BATCH", 1000);
}

template <typename BatchConsumer>
void generate_streamlines_parallel(size_t streamlines,
                                   LengthProfile profile,
                                   size_t threads,
                                   size_t batch_size,
                                   uint32_t base_seed,
                                   BatchConsumer consumer) {
  const size_t total_batches = (streamlines + batch_size - 1) / batch_size;
  std::atomic<size_t> next_batch{0};
  std::mutex mutex;
  std::condition_variable cv;
  std::map<size_t, std::vector<std::vector<std::array<float, 3>>>> completed;
  std::condition_variable cv_producer;
  size_t inflight_batches = 0;
  const size_t max_inflight = std::max<size_t>(1, parse_env_size("TRX_BENCH_QUEUE_MAX", 8));

  auto worker = [&]() {
    for (;;) {
      size_t batch_idx;
      {
        // Wait for queue space BEFORE grabbing batch index to avoid missed notifications
        std::unique_lock<std::mutex> lock(mutex);
        cv_producer.wait(lock, [&]() { return inflight_batches < max_inflight || next_batch.load() >= total_batches; });
        batch_idx = next_batch.fetch_add(1);
        if (batch_idx >= total_batches) {
          return;
        }
        ++inflight_batches;
      }
      const size_t start = batch_idx * batch_size;
      const size_t count = std::min(batch_size, streamlines - start);
      std::vector<std::vector<std::array<float, 3>>> batch;
      batch.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        const uint32_t seed = base_seed + static_cast<uint32_t>(start + i);
        batch.push_back(generate_streamline_points_seeded(seed, profile));
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        completed.emplace(batch_idx, std::move(batch));
      }
      cv.notify_one();
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (size_t t = 0; t < threads; ++t) {
    workers.emplace_back(worker);
  }

  for (size_t expected = 0; expected < total_batches; ++expected) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&]() { return completed.find(expected) != completed.end(); });
    auto batch = std::move(completed[expected]);
    completed.erase(expected);
    if (inflight_batches > 0) {
      --inflight_batches;
    }
    lock.unlock();
    cv_producer.notify_all();  // Wake all waiting workers, not just one, to avoid deadlock

    const size_t start = expected * batch_size;
    consumer(start, batch);
  }

  for (auto &worker_thread : workers) {
    worker_thread.join();
  }
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
  LengthProfile profile = LengthProfile::Mixed;
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
      << "\"length_profile\":" << static_cast<int>(scenario.profile) << ","
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
                                LengthProfile profile,
                                bool add_dps,
                                bool add_dpv,
                                zip_uint32_t compression) {
  trx::TrxStream stream("float16");
  stream.set_metadata_mode(trx::TrxStream::MetadataMode::OnDisk);
  
  // Scale metadata buffer with TRX_BENCH_BUFFER_MULTIPLIER for slow storage
  const size_t buffer_multiplier = std::max<size_t>(1, parse_env_size("TRX_BENCH_BUFFER_MULTIPLIER", 1));
  stream.set_metadata_buffer_max_bytes(64ULL * 1024ULL * 1024ULL * buffer_multiplier);
  stream.set_positions_buffer_max_bytes(buffer_bytes_for_streamlines(streamlines));

  const size_t threads = bench_threads();
  const size_t batch_size = std::max<size_t>(1, bench_batch_size());
  const uint32_t base_seed = static_cast<uint32_t>(1337 + streamlines + static_cast<size_t>(profile) * 13);
  const size_t progress_every = parse_env_size("TRX_BENCH_LOG_PROGRESS_EVERY", 0);
  log_bench_config("file_size_generate", threads, batch_size);

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

  std::vector<float> dps;
  std::vector<float> dpv;
  if (add_dps) {
    dps.reserve(streamlines);
  }
  if (add_dpv) {
    const size_t estimated_vertices = streamlines * estimate_points_per_streamline(profile);
    dpv.reserve(estimated_vertices);
  }

  generate_streamlines_parallel(
      streamlines,
      profile,
      threads,
      batch_size,
      base_seed,
      [&](size_t start, const std::vector<std::vector<std::array<float, 3>>> &batch) {
        if (parse_env_bool("TRX_BENCH_LOG", false)) {
          std::cerr << "[trx-bench] batch file_size start=" << start << " count=" << batch.size() << std::endl;
        }
        for (size_t i = 0; i < batch.size(); ++i) {
          const auto &points = batch[i];
          stream.push_streamline(points);
          if (add_dps) {
            dps.push_back(1.0f);
          }
          if (add_dpv) {
            dpv.insert(dpv.end(), points.size(), 0.5f);
          }
          const size_t global_idx = start + i + 1;
          if (progress_every > 0 && (global_idx % progress_every == 0)) {
            std::cerr << "[trx-bench] progress file_size streamlines=" << global_idx << " / " << streamlines
                      << std::endl;
          }
          if (collect_rss && sample_every > 0 && (global_idx % sample_every == 0)) {
            record_sample("generate");
          }
        }
      });

  if (add_dps) {
    stream.push_dps_from_vector("dps_scalar", "float32", dps);
  }
  if (add_dpv) {
    stream.push_dpv_from_vector("dpv_scalar", "float32", dpv);
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

  const auto start = std::chrono::steady_clock::now();
  stream.finalize<half>(out_path, compression);
  const auto end = std::chrono::steady_clock::now();

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
    scenario.profile = profile;
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

TrxOnDisk build_trx_file_on_disk_single(size_t streamlines,
                                        GroupScenario scenario,
                                        bool add_dps,
                                        bool add_dpv,
                                        LengthProfile profile,
                                        zip_uint32_t compression,
                                        const std::string &out_path_override = "",
                                        bool finalize_to_directory = false) {
  trx::TrxStream stream("float16");
  stream.set_metadata_mode(trx::TrxStream::MetadataMode::OnDisk);
  
  // Scale buffers with TRX_BENCH_BUFFER_MULTIPLIER for slow storage
  const size_t buffer_multiplier = std::max<size_t>(1, parse_env_size("TRX_BENCH_BUFFER_MULTIPLIER", 1));
  stream.set_metadata_buffer_max_bytes(64ULL * 1024ULL * 1024ULL * buffer_multiplier);
  stream.set_positions_buffer_max_bytes(buffer_bytes_for_streamlines(streamlines));
  const size_t progress_every = parse_env_size("TRX_BENCH_LOG_PROGRESS_EVERY", 0);

  const auto group_count = group_count_for(scenario);
  const auto &group_names = group_names_for(scenario);
  std::vector<std::vector<uint32_t>> groups(group_count);

  const size_t threads = bench_threads();
  const size_t batch_size = std::max<size_t>(1, bench_batch_size());
  const uint32_t base_seed = static_cast<uint32_t>(1337 + streamlines + static_cast<size_t>(scenario) * 31);
  log_bench_config("build_trx_generate", threads, batch_size);

  std::vector<float> dps;
  std::vector<float> dpv;
  if (add_dps) {
    dps.reserve(streamlines);
  }
  if (add_dpv) {
    const size_t estimated_vertices = streamlines * estimate_points_per_streamline(profile);
    dpv.reserve(estimated_vertices);
  }

  size_t total_vertices = 0;
  generate_streamlines_parallel(
      streamlines,
      profile,
      threads,
      batch_size,
      base_seed,
      [&](size_t start, const std::vector<std::vector<std::array<float, 3>>> &batch) {
        if (parse_env_bool("TRX_BENCH_LOG", false)) {
          std::cerr << "[trx-bench] batch build_trx start=" << start << " count=" << batch.size() << std::endl;
        }
        for (size_t i = 0; i < batch.size(); ++i) {
          const auto &points = batch[i];
          total_vertices += points.size();
          stream.push_streamline(points);
          if (add_dps) {
            dps.push_back(1.0f);
          }
          if (add_dpv) {
            dpv.insert(dpv.end(), points.size(), 0.5f);
          }
          const size_t global_idx = start + i;
          if (group_count > 0) {
            groups[global_idx % group_count].push_back(static_cast<uint32_t>(global_idx));
          }
        if (progress_every > 0 && ((global_idx + 1) % progress_every == 0)) {
          if (parse_env_bool("TRX_BENCH_CHILD_LOG", false) || parse_env_bool("TRX_BENCH_LOG", false)) {
            const char *shard_env = std::getenv("TRX_BENCH_SHARD_INDEX");
            const std::string shard_prefix = shard_env ? std::string(" shard=") + shard_env : "";
            std::cerr << "[trx-bench] progress build_trx" << shard_prefix << " streamlines=" << (global_idx + 1)
                      << " / " << streamlines << std::endl;
          }
        }
        }
      });

  if (add_dps) {
    stream.push_dps_from_vector("dps_scalar", "float32", dps);
  }
  if (add_dpv) {
    stream.push_dpv_from_vector("dpv_scalar", "float32", dpv);
  }
  if (group_count > 0) {
    for (size_t g = 0; g < group_count; ++g) {
      stream.push_group_from_indices(group_names[g], groups[g]);
    }
  }

  const std::string out_path = out_path_override.empty() ? make_temp_path("trx_input") : out_path_override;
  if (finalize_to_directory) {
    // Use persistent variant to avoid removing pre-created shard directories
    stream.finalize_directory_persistent(out_path);
  } else {
    stream.finalize<half>(out_path, compression);
  }
  if (out_path_override.empty() && !finalize_to_directory) {
    register_cleanup(out_path);
  }
  return {out_path, streamlines, total_vertices, 0.0, 1};
}

void build_trx_shard(const std::string &out_path,
                     size_t streamlines,
                     GroupScenario scenario,
                     bool add_dps,
                     bool add_dpv,
                     LengthProfile profile,
                     zip_uint32_t compression) {
  (void)build_trx_file_on_disk_single(streamlines,
                                      scenario,
                                      add_dps,
                                      add_dpv,
                                      profile,
                                      compression,
                                      out_path,
                                      true);
  
  // Defensive validation: ensure all required files were written by finalize_directory_persistent
  std::error_code ec;
  const auto header_path = trx::fs::path(out_path) / "header.json";
  if (!trx::fs::exists(header_path, ec)) {
    throw std::runtime_error("Shard missing header.json after finalize_directory_persistent: " + header_path.string());
  }
  const auto positions_path = find_file_by_prefix(out_path, "positions.");
  if (positions_path.empty()) {
    throw std::runtime_error("Shard missing positions after finalize_directory_persistent: " + out_path);
  }
  const auto offsets_path = find_file_by_prefix(out_path, "offsets.");
  if (offsets_path.empty()) {
    throw std::runtime_error("Shard missing offsets after finalize_directory_persistent: " + out_path);
  }
  const auto ok_path = trx::fs::path(out_path) / "SHARD_OK";
  std::ofstream ok(ok_path, std::ios::out | std::ios::trunc);
  if (ok.is_open()) {
    ok << "ok\n";
    ok.flush();
    ok.close();
  }
  
  // Force filesystem sync to ensure all shard data is visible to parent process
#if defined(__unix__) || defined(__APPLE__)
  sync();
  // Brief sleep to ensure filesystem metadata updates are visible across processes
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif
}

TrxOnDisk build_trx_file_on_disk(size_t streamlines,
                                 GroupScenario scenario,
                                 bool add_dps,
                                 bool add_dpv,
                                 LengthProfile profile,
                                 zip_uint32_t compression) {
  size_t processes = parse_env_size("TRX_BENCH_PROCESSES", 1);
  const size_t mp_min_streamlines = parse_env_size("TRX_BENCH_MP_MIN_STREAMLINES", 1000000);
  if (streamlines < mp_min_streamlines) {
    processes = 1;
  }
  if (processes <= 1) {
    return build_trx_file_on_disk_single(streamlines, scenario, add_dps, add_dpv, profile, compression);
  }
#if defined(__unix__) || defined(__APPLE__)
  g_cleanup_owner_pid = getpid();
  const std::string shard_root = make_work_dir_name("trx_shards");
  {
    std::error_code ec;
    trx::fs::create_directories(shard_root, ec);
    if (ec) {
      throw std::runtime_error("Failed to create shard root: " + shard_root);
    }
  }
  {
    const std::string marker = shard_root + trx::SEPARATOR + "SHARD_ROOT_CREATED";
    std::ofstream out(marker, std::ios::out | std::ios::trunc);
    if (out.is_open()) {
      out << "ok\n";
      out.flush();
      out.close();
    }
  }
  if (parse_env_bool("TRX_BENCH_LOG", false)) {
    std::cerr << "[trx-bench] shard_root " << shard_root << std::endl;
  }
  std::vector<size_t> counts(processes, streamlines / processes);
  const size_t remainder = streamlines % processes;
  for (size_t i = 0; i < remainder; ++i) {
    counts[i] += 1;
  }

              std::vector<std::string> shard_paths(processes);
              std::vector<std::string> status_paths(processes);
              for (size_t i = 0; i < processes; ++i) {
                shard_paths[i] = shard_root + trx::SEPARATOR + "shard_" + std::to_string(i);
                status_paths[i] = shard_root + trx::SEPARATOR + "shard_" + std::to_string(i) + ".status";
                
                // Pre-create shard directories to validate filesystem writability before forking.
                // finalize_directory_persistent() will use these existing directories without
                // removing them, avoiding race conditions in the multiprocess workflow.
                std::error_code ec;
                trx::fs::create_directories(shard_paths[i], ec);
                if (ec) {
                  throw std::runtime_error("Failed to create shard dir: " + shard_paths[i] + " " + ec.message());
                }
                std::ofstream status(status_paths[i], std::ios::out | std::ios::trunc);
                if (status.is_open()) {
                  status << "pending\n";
                }
              }
              if (parse_env_bool("TRX_BENCH_LOG", false)) {
                for (size_t i = 0; i < processes; ++i) {
                  std::cerr << "[trx-bench] shard_path[" << i << "] " << shard_paths[i] << std::endl;
                }
              }

  std::vector<pid_t> pids;
  pids.reserve(processes);
  for (size_t i = 0; i < processes; ++i) {
    const pid_t pid = fork();
    if (pid == 0) {
      try {
        setenv("TRX_BENCH_THREADS", "1", 1);
        setenv("TRX_BENCH_BATCH", "1000", 1);
        setenv("TRX_BENCH_LOG", "0", 1);
        setenv("TRX_BENCH_SHARD_INDEX", std::to_string(i).c_str(), 1);
        if (parse_env_bool("TRX_BENCH_LOG", false)) {
          std::cerr << "[trx-bench] shard_child_start path=" << shard_paths[i] << std::endl;
        }
        {
          std::ofstream status(status_paths[i], std::ios::out | std::ios::trunc);
          if (status.is_open()) {
            status << "started pid=" << getpid() << "\n";
            status.flush();
          }
        }
        build_trx_shard(shard_paths[i], counts[i], scenario, add_dps, add_dpv, profile, compression);
        {
          std::ofstream status(status_paths[i], std::ios::out | std::ios::trunc);
          if (status.is_open()) {
            status << "ok\n";
            status.flush();
          }
        }
        _exit(0);
      } catch (const std::exception &ex) {
        std::ofstream out(status_paths[i], std::ios::out | std::ios::trunc);
        if (out.is_open()) {
          out << ex.what() << "\n";
          out.flush();
          out.close();
        }
        _exit(1);
      } catch (...) {
        std::ofstream out(status_paths[i], std::ios::out | std::ios::trunc);
        if (out.is_open()) {
          out << "Unknown error\n";
          out.flush();
          out.close();
        }
        _exit(1);
      }
    }
    if (pid < 0) {
      throw std::runtime_error("Failed to fork shard process");
    }
    pids.push_back(pid);
  }

  for (size_t i = 0; i < pids.size(); ++i) {
    const auto pid = pids[i];
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::string detail;
      std::ifstream in(status_paths[i]);
      if (in.is_open()) {
        std::getline(in, detail);
      }
      if (detail.empty()) {
        detail = "No status file content";
      }
      throw std::runtime_error("Shard process failed: " + detail);
    }
  }

  const size_t shard_wait_ms = parse_env_size("TRX_BENCH_SHARD_WAIT_MS", 10000);
  wait_for_shard_ok(shard_paths, status_paths, shard_wait_ms);

  size_t total_vertices = 0;
  size_t total_streamlines = 0;
  std::vector<size_t> shard_vertices(processes, 0);
  std::vector<size_t> shard_streamlines(processes, 0);
  for (size_t i = 0; i < processes; ++i) {
    const auto ok_path = trx::fs::path(shard_paths[i]) / "SHARD_OK";
    std::error_code ok_ec;
    if (!trx::fs::exists(ok_path, ok_ec)) {
      std::string detail;
      std::ifstream in(status_paths[i]);
      if (in.is_open()) {
        std::getline(in, detail);
      }
      if (detail.empty()) {
        detail = "SHARD_OK missing for " + shard_paths[i];
      }
      throw std::runtime_error("Shard process failed: " + detail);
    }
    std::error_code ec;
    if (!trx::fs::exists(shard_paths[i], ec) || !trx::fs::is_directory(shard_paths[i], ec)) {
      const auto root_files = list_files(shard_root);
      std::string detail = "Shard output directory missing: " + shard_paths[i];
      if (!root_files.empty()) {
        detail += " root_files=[";
        for (size_t j = 0; j < root_files.size(); ++j) {
          if (j > 0) {
            detail += ",";
          }
          detail += root_files[j];
        }
        detail += "]";
      }
      throw std::runtime_error(detail);
    }
    const auto header_path = trx::fs::path(shard_paths[i]) / "header.json";
    if (!trx::fs::exists(header_path, ec)) {
      const auto files = list_files(shard_paths[i]);
      std::string detail = "Shard missing header.json: " + header_path.string();
      if (!files.empty()) {
        detail += " files=[";
        for (size_t j = 0; j < files.size(); ++j) {
          if (j > 0) {
            detail += ",";
          }
          detail += files[j];
        }
        detail += "]";
      }
      const auto root_files = list_files(shard_root);
      if (!root_files.empty()) {
        detail += " root_files=[";
        for (size_t j = 0; j < root_files.size(); ++j) {
          if (j > 0) {
            detail += ",";
          }
          detail += root_files[j];
        }
        detail += "]";
      }
      throw std::runtime_error(detail);
    }
    const auto counts = read_header_counts(shard_paths[i]);
    shard_streamlines[i] = counts.first;
    shard_vertices[i] = counts.second;
    total_streamlines += counts.first;
    total_vertices += counts.second;
  }

              const auto merge_start = std::chrono::steady_clock::now();
              const auto group_count = group_count_for(scenario);
              const auto &group_names = group_names_for(scenario);

              const std::string merge_dir = make_temp_dir_path("trx_merge");
              const auto shard_positions0 = find_file_by_prefix(shard_paths[0], "positions.");
              const auto shard_offsets0 = find_file_by_prefix(shard_paths[0], "offsets.");
              if (shard_positions0.empty()) {
                throw std::runtime_error("Missing positions file in first shard: " + shard_paths[0]);
              }
              if (shard_offsets0.empty()) {
                throw std::runtime_error("Missing offsets file in first shard: " + shard_paths[0]);
              }
              const auto positions_filename = trx::fs::path(shard_positions0).filename().string();
              const auto offsets_filename = trx::fs::path(shard_offsets0).filename().string();
              const auto positions_path = trx::fs::path(merge_dir) / positions_filename;
              const auto offsets_path = trx::fs::path(merge_dir) / offsets_filename;

              {
                std::ofstream out_pos(positions_path, std::ios::binary | std::ios::out | std::ios::trunc);
                if (!out_pos.is_open()) {
                  throw std::runtime_error("Failed to open output positions file: " + positions_path.string());
                }
              }
              {
                std::ofstream out_off(offsets_path, std::ios::binary | std::ios::out | std::ios::trunc);
                if (!out_off.is_open()) {
                  throw std::runtime_error("Failed to open output offsets file: " + offsets_path.string());
                }
              }

              std::vector<std::string> dps_files;
              std::vector<std::string> dpv_files;
              if (add_dps) {
                dps_files = list_files((trx::fs::path(shard_paths[0]) / "dps").string());
                if (dps_files.empty()) {
                  throw std::runtime_error("No DPS files found in shard: " + shard_paths[0]);
                }
              }
              if (add_dpv) {
                dpv_files = list_files((trx::fs::path(shard_paths[0]) / "dpv").string());
                if (dpv_files.empty()) {
                  throw std::runtime_error("No DPV files found in shard: " + shard_paths[0]);
                }
              }
              std::vector<std::string> group_files;
              if (group_count > 0) {
                group_files = list_files((trx::fs::path(shard_paths[0]) / "groups").string());
                if (group_files.empty()) {
                  throw std::runtime_error("No group files found in shard: " + shard_paths[0]);
                }
              }

              if (add_dps) {
                trx::fs::create_directories(trx::fs::path(merge_dir) / "dps");
                for (const auto &name : dps_files) {
                  const auto dst = trx::fs::path(merge_dir) / "dps" / name;
                  std::ofstream out(dst, std::ios::binary | std::ios::out | std::ios::trunc);
                  if (!out.is_open()) {
                    throw std::runtime_error("Failed to create DPS file: " + dst.string());
                  }
                }
              }
              if (add_dpv) {
                trx::fs::create_directories(trx::fs::path(merge_dir) / "dpv");
                for (const auto &name : dpv_files) {
                  const auto dst = trx::fs::path(merge_dir) / "dpv" / name;
                  std::ofstream out(dst, std::ios::binary | std::ios::out | std::ios::trunc);
                  if (!out.is_open()) {
                    throw std::runtime_error("Failed to create DPV file: " + dst.string());
                  }
                }
              }
              if (group_count > 0) {
                trx::fs::create_directories(trx::fs::path(merge_dir) / "groups");
                for (const auto &name : group_files) {
                  const auto dst = trx::fs::path(merge_dir) / "groups" / name;
                  std::ofstream out(dst, std::ios::binary | std::ios::out | std::ios::trunc);
                  if (!out.is_open()) {
                    throw std::runtime_error("Failed to create group file: " + dst.string());
                  }
                }
              }

              size_t vertex_offset = 0;
              size_t streamline_offset = 0;
              for (size_t i = 0; i < processes; ++i) {
                const auto shard_dir = shard_paths[i];
                const auto shard_positions = find_file_by_prefix(shard_dir, "positions.");
                const auto shard_offsets = find_file_by_prefix(shard_dir, "offsets.");
                if (shard_positions.empty()) {
                  throw std::runtime_error("Missing positions file in shard: " + shard_dir);
                }
                if (shard_offsets.empty()) {
                  throw std::runtime_error("Missing offsets file in shard: " + shard_dir);
                }

                copy_file_append(shard_positions, positions_path.string());

                {
                  const bool offsets_u32 = offsets_filename.find("uint32") != std::string::npos;
                  std::ifstream in(shard_offsets, std::ios::binary);
                  if (!in.is_open()) {
                    throw std::runtime_error("Failed to open shard offsets: " + shard_offsets);
                  }
                  std::ofstream out(offsets_path, std::ios::binary | std::ios::out | std::ios::app);
                  if (!out.is_open()) {
                    throw std::runtime_error("Failed to open output offsets file: " + offsets_path.string());
                  }
                  constexpr size_t kBatch = 1 << 14;
                  const bool skip_first_value = (i != 0);
                  bool skipped_first = false;
                  if (offsets_u32) {
                    std::vector<uint32_t> buffer(kBatch);
                    while (in) {
                      in.read(reinterpret_cast<char *>(buffer.data()),
                              static_cast<std::streamsize>(buffer.size() * sizeof(uint32_t)));
                      const std::streamsize count = in.gcount();
                      if (count <= 0) {
                        break;
                      }
                      const size_t elems = static_cast<size_t>(count) / sizeof(uint32_t);
                      size_t start = 0;
                      if (skip_first_value && !skipped_first) {
                        start = 1;
                        skipped_first = true;
                      }
                      for (size_t j = start; j < elems; ++j) {
                        const uint64_t value = static_cast<uint64_t>(buffer[j]) + static_cast<uint64_t>(vertex_offset);
                        if (value > std::numeric_limits<uint32_t>::max()) {
                          throw std::runtime_error("Offsets overflow uint32 during merge.");
                        }
                        buffer[j] = static_cast<uint32_t>(value);
                      }
                      if (elems > start) {
                        out.write(reinterpret_cast<const char *>(buffer.data() + start),
                                  static_cast<std::streamsize>((elems - start) * sizeof(uint32_t)));
                      }
                    }
                  } else {
                    std::vector<uint64_t> buffer(kBatch);
                    while (in) {
                      in.read(reinterpret_cast<char *>(buffer.data()),
                              static_cast<std::streamsize>(buffer.size() * sizeof(uint64_t)));
                      const std::streamsize count = in.gcount();
                      if (count <= 0) {
                        break;
                      }
                      const size_t elems = static_cast<size_t>(count) / sizeof(uint64_t);
                      size_t start = 0;
                      if (skip_first_value && !skipped_first) {
                        start = 1;
                        skipped_first = true;
                      }
                      for (size_t j = start; j < elems; ++j) {
                        buffer[j] += static_cast<uint64_t>(vertex_offset);
                      }
                      if (elems > start) {
                        out.write(reinterpret_cast<const char *>(buffer.data() + start),
                                  static_cast<std::streamsize>((elems - start) * sizeof(uint64_t)));
                      }
                    }
                  }
                }

                if (add_dps) {
                  const auto shard_dps = trx::fs::path(shard_dir) / "dps";
                  for (const auto &name : dps_files) {
                    const auto src = shard_dps / name;
                    const auto dst = trx::fs::path(merge_dir) / "dps" / name;
                    if (!trx::fs::exists(src)) {
                      throw std::runtime_error("Missing DPS file in shard: " + src.string());
                    }
                    copy_file_append(src.string(), dst.string());
                  }
                }

                if (add_dpv) {
                  const auto shard_dpv = trx::fs::path(shard_dir) / "dpv";
                  for (const auto &name : dpv_files) {
                    const auto src = shard_dpv / name;
                    const auto dst = trx::fs::path(merge_dir) / "dpv" / name;
                    if (!trx::fs::exists(src)) {
                      throw std::runtime_error("Missing DPV file in shard: " + src.string());
                    }
                    copy_file_append(src.string(), dst.string());
                  }
                }

                if (group_count > 0) {
                  const auto shard_groups = trx::fs::path(shard_dir) / "groups";
                  for (const auto &name : group_files) {
                    const auto src = shard_groups / name;
                    const auto dst = trx::fs::path(merge_dir) / "groups" / name;
                    if (!trx::fs::exists(src)) {
                      throw std::runtime_error("Missing group file in shard: " + src.string());
                    }
                    std::ifstream in(src, std::ios::binary);
                    if (!in.is_open()) {
                      throw std::runtime_error("Failed to open shard group: " + src.string());
                    }
                    std::ofstream out(dst, std::ios::binary | std::ios::out | std::ios::app);
                    if (!out.is_open()) {
                      throw std::runtime_error("Failed to open output group file: " + dst.string());
                    }
                    constexpr size_t kBatch = 1 << 14;
                    std::vector<uint32_t> buffer(kBatch);
                    while (in) {
                      in.read(reinterpret_cast<char *>(buffer.data()),
                              static_cast<std::streamsize>(buffer.size() * sizeof(uint32_t)));
                      const std::streamsize count = in.gcount();
                      if (count <= 0) {
                        break;
                      }
                      const size_t elems = static_cast<size_t>(count) / sizeof(uint32_t);
                      for (size_t j = 0; j < elems; ++j) {
                        buffer[j] += static_cast<uint32_t>(streamline_offset);
                      }
                      out.write(reinterpret_cast<const char *>(buffer.data()),
                                static_cast<std::streamsize>(elems * sizeof(uint32_t)));
                    }
                  }
                }

                vertex_offset += shard_vertices[i];
                streamline_offset += shard_streamlines[i];
              }

              // Read header before cleanup to avoid accessing deleted files
              const json header_json = read_header_json(shard_paths[0]);
              json::object header_obj = header_json.object_items();
              header_obj["NB_VERTICES"] = json(static_cast<double>(total_vertices));
              header_obj["NB_STREAMLINES"] = json(static_cast<double>(total_streamlines));
              const json header = header_obj;
              {
                const auto header_path = trx::fs::path(merge_dir) / "header.json";
                std::ofstream out(header_path, std::ios::out | std::ios::trunc);
                if (!out.is_open()) {
                  throw std::runtime_error("Failed to write header.json: " + header_path.string());
                }
                out << header.dump();
              }

              const std::string zip_path = make_temp_path("trx_input");
              int errorp;
              zip_t *zf = zip_open(zip_path.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp);
              if (zf == nullptr) {
                throw std::runtime_error("Could not open archive " + zip_path + ": " + strerror(errorp));
              }
              const std::string header_payload = header.dump() + "\n";
              zip_source_t *header_source =
                  zip_source_buffer(zf, header_payload.data(), header_payload.size(), 0 /* do not free */);
              if (header_source == nullptr) {
                zip_close(zf);
                throw std::runtime_error("Failed to create zip source for header.json: " + std::string(zip_strerror(zf)));
              }
              const zip_int64_t header_idx = zip_file_add(zf, "header.json", header_source, ZIP_FL_ENC_UTF_8 | ZIP_FL_OVERWRITE);
              if (header_idx < 0) {
                zip_source_free(header_source);
                zip_close(zf);
                throw std::runtime_error("Failed to add header.json to archive: " + std::string(zip_strerror(zf)));
              }
              const zip_int32_t compression_mode = static_cast<zip_int32_t>(compression);
              if (zip_set_file_compression(zf, header_idx, compression_mode, 0) < 0) {
                zip_close(zf);
                throw std::runtime_error("Failed to set compression for header.json: " + std::string(zip_strerror(zf)));
              }
              const std::unordered_set<std::string> skip = {"header.json"};
              trx::zip_from_folder(zf, merge_dir, merge_dir, compression, &skip);
              if (zip_close(zf) != 0) {
                throw std::runtime_error("Unable to close archive " + zip_path + ": " + zip_strerror(zf));
              }
              trx::fs::remove_all(merge_dir);
              const std::string out_path = zip_path;

              register_cleanup(out_path);
              const auto merge_end = std::chrono::steady_clock::now();
              const std::chrono::duration<double, std::milli> merge_elapsed = merge_end - merge_start;
              
              // Final cleanup of shard directories after merge is complete
              if (!parse_env_bool("TRX_BENCH_KEEP_SHARDS", false)) {
                std::error_code ec;
                trx::fs::remove_all(shard_root, ec);
              }
              return {out_path, streamlines, total_vertices, merge_elapsed.count(), processes};
#else
  (void)processes;
  return build_trx_file_on_disk_single(streamlines, scenario, add_dps, add_dpv, profile, compression);
#endif
}

struct QueryDataset {
  std::unique_ptr<trx::TrxFile<half>> trx;
  std::vector<std::array<half, 6>> aabbs;
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
  LengthProfile profile = LengthProfile::Mixed;
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
  const auto profile = static_cast<LengthProfile>(state.range(1));
  const bool add_dps = state.range(2) != 0;
  const bool add_dpv = state.range(3) != 0;
  const bool use_zip = state.range(4) != 0;
  const auto compression = use_zip ? ZIP_CM_DEFLATE : ZIP_CM_STORE;
  const size_t skip_zip_at = parse_env_size("TRX_BENCH_SKIP_ZIP_AT", 5000000);
  if (use_zip && streamlines >= skip_zip_at) {
    state.SkipWithMessage("zip compression skipped for large streamlines");
    return;
  }
  log_bench_start("BM_TrxFileSize_Float16",
                  "streamlines=" + std::to_string(streamlines) + " profile=" + std::to_string(state.range(1)) +
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
        build_trx_file_on_disk(streamlines, GroupScenario::None, add_dps, add_dpv, profile, compression);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    total_build_ms += elapsed.count();
    total_merge_ms += on_disk.shard_merge_ms;
    total_merge_processes += static_cast<double>(on_disk.shard_processes);
    total_write_ms += elapsed.count();
    total_file_bytes += static_cast<double>(file_size_bytes(on_disk.path));
  }

  state.counters["streamlines"] = static_cast<double>(streamlines);
  state.counters["length_profile"] = static_cast<double>(state.range(1));
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
                "streamlines=" + std::to_string(streamlines) + " profile=" + std::to_string(state.range(1)));
}

static void BM_TrxStream_TranslateWrite(benchmark::State &state) {
  const size_t streamlines = static_cast<size_t>(state.range(0));
  const auto scenario = static_cast<GroupScenario>(state.range(1));
  const bool add_dps = state.range(2) != 0;
  const bool add_dpv = state.range(3) != 0;
  const size_t progress_every = parse_env_size("TRX_BENCH_LOG_PROGRESS_EVERY", 0);
  log_bench_config("translate_write", bench_threads(), std::max<size_t>(1, bench_batch_size()));
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
                  build_trx_file_on_disk(streamlines, scenario, add_dps, add_dpv, LengthProfile::Mixed, ZIP_CM_STORE));
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
    const auto out_info = trx::prepare_positions_output(trx, out_dir);

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
    int errorp;
    zip_t *zf = zip_open(out_path.c_str(), ZIP_CREATE + ZIP_TRUNCATE, &errorp);
    if (zf == nullptr) {
      trx::rm_dir(out_dir);
      throw std::runtime_error("Could not open archive " + out_path + ": " + strerror(errorp));
    }
    trx::zip_from_folder(zf, out_dir, out_dir, ZIP_CM_STORE, nullptr);
    if (zip_close(zf) != 0) {
      trx::rm_dir(out_dir);
      throw std::runtime_error("Unable to close archive " + out_path + ": " + zip_strerror(zf));
    }
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
  state.counters["length_profile"] = static_cast<double>(static_cast<int>(LengthProfile::Mixed));
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
    auto on_disk = build_trx_file_on_disk(streamlines, scenario, add_dps, add_dpv, LengthProfile::Mixed, ZIP_CM_STORE);
    dataset.trx = trx::load<half>(on_disk.path);
    dataset.aabbs = dataset.trx->build_streamline_aabbs();
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
      auto subset = dataset.trx->query_aabb(min_corner, max_corner, &dataset.aabbs);
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
    params.profile = LengthProfile::Mixed;
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
  const std::array<int, 3> profiles = {static_cast<int>(LengthProfile::Short),
                                       static_cast<int>(LengthProfile::Medium),
                                       static_cast<int>(LengthProfile::Long)};
  const std::array<int, 2> flags = {0, 1};
  const auto counts_desc = streamlines_for_benchmarks();
  for (const auto count : counts_desc) {
    for (const auto profile : profiles) {
      for (const auto dps : flags) {
        for (const auto dpv : flags) {
          for (const auto compression : flags) {
            bench->Args({static_cast<int64_t>(count), profile, dps, dpv, compression});
          }
        }
      }
    }
  }
}

static void ApplyStreamArgs(benchmark::internal::Benchmark *bench) {
  const std::array<int, 3> groups = {static_cast<int>(GroupScenario::None),
                                     static_cast<int>(GroupScenario::Bundles),
                                     static_cast<int>(GroupScenario::Connectome)};
  const std::array<int, 2> flags = {0, 1};
  const auto counts_desc = streamlines_for_benchmarks();
  for (const auto count : counts_desc) {
    for (const auto group_case : groups) {
      for (const auto dps : flags) {
        for (const auto dpv : flags) {
          bench->Args({static_cast<int64_t>(count), group_case, dps, dpv});
        }
      }
    }
  }
}

static void ApplyQueryArgs(benchmark::internal::Benchmark *bench) {
  const std::array<int, 3> groups = {static_cast<int>(GroupScenario::None),
                                     static_cast<int>(GroupScenario::Bundles),
                                     static_cast<int>(GroupScenario::Connectome)};
  const std::array<int, 2> flags = {0, 1};
  const auto counts_desc = streamlines_for_benchmarks();
  for (const auto count : counts_desc) {
    for (const auto group_case : groups) {
      for (const auto dps : flags) {
        for (const auto dpv : flags) {
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
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
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
