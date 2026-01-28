#ifndef TRX_STREAMLINES_OPS_H
#define TRX_STREAMLINES_OPS_H

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace trxmmap {
constexpr int kMinNbPoints = 5;

inline float round_to_precision(float value, int precision) {
  if (precision < 0) {
    return value;
  }
  const float scale = std::pow(10.0f, static_cast<float>(precision));
  return std::round(value * scale) / scale;
}

inline std::string get_streamline_key(const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> &streamline,
                                      int precision = -1) {
  const Eigen::Index rows = streamline.rows();
  std::vector<float> key_data;
  if (rows < kMinNbPoints) {
    key_data.reserve(static_cast<size_t>(rows * 3));
    for (Eigen::Index i = 0; i < rows; ++i) {
      for (Eigen::Index j = 0; j < 3; ++j) {
        key_data.push_back(round_to_precision(streamline(i, j), precision));
      }
    }
  } else {
    key_data.reserve(static_cast<size_t>(kMinNbPoints * 2 * 3));
    for (int i = 0; i < kMinNbPoints; ++i) {
      for (int j = 0; j < 3; ++j) {
        key_data.push_back(round_to_precision(streamline(i, j), precision));
      }
    }
    for (Eigen::Index i = rows - kMinNbPoints; i < rows; ++i) {
      for (int j = 0; j < 3; ++j) {
        key_data.push_back(round_to_precision(streamline(i, j), precision));
      }
    }
  }

  std::string key;
  key.resize(key_data.size() * sizeof(float));
  if (!key.empty()) {
    std::memcpy(key.data(), key_data.data(), key.size());
  }
  return key;
}

using StreamlineKeyMap = std::unordered_map<std::string, uint32_t>;

inline StreamlineKeyMap
hash_streamlines(const std::vector<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> &streamlines,
                 uint32_t start_index = 0,
                 int precision = -1) {
  StreamlineKeyMap result;
  result.reserve(streamlines.size());
  for (size_t i = 0; i < streamlines.size(); ++i) {
    const auto key = get_streamline_key(streamlines[i], precision);
    result[key] = static_cast<uint32_t>(start_index + i);
  }
  return result;
}

inline StreamlineKeyMap intersection(const StreamlineKeyMap &left, const StreamlineKeyMap &right) {
  StreamlineKeyMap result;
  for (const auto &item : left) {
    if (right.find(item.first) != right.end()) {
      result[item.first] = item.second;
    }
  }
  return result;
}

inline StreamlineKeyMap difference(const StreamlineKeyMap &left, const StreamlineKeyMap &right) {
  StreamlineKeyMap result;
  for (const auto &item : left) {
    if (right.find(item.first) == right.end()) {
      result[item.first] = item.second;
    }
  }
  return result;
}

inline StreamlineKeyMap union_maps(const StreamlineKeyMap &left, const StreamlineKeyMap &right) {
  StreamlineKeyMap result = right;
  for (const auto &item : left) {
    result[item.first] = item.second;
  }
  return result;
}

inline std::pair<std::vector<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>, std::vector<uint32_t>>
perform_streamlines_operation(
    const std::function<StreamlineKeyMap(const StreamlineKeyMap &, const StreamlineKeyMap &)> &operation,
    const std::vector<std::vector<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>> &streamlines,
    int precision = 0) {
  std::vector<uint32_t> offsets;
  offsets.reserve(streamlines.size());
  uint32_t acc = 0;
  for (size_t i = 0; i < streamlines.size(); ++i) {
    offsets.push_back(acc);
    acc += static_cast<uint32_t>(streamlines[i].size());
  }

  std::vector<StreamlineKeyMap> hashes;
  hashes.reserve(streamlines.size());
  for (size_t i = 0; i < streamlines.size(); ++i) {
    hashes.push_back(hash_streamlines(streamlines[i], offsets[i], precision));
  }

  if (hashes.empty()) {
    return {{}, {}};
  }

  StreamlineKeyMap to_keep = hashes[0];
  for (size_t i = 1; i < hashes.size(); ++i) {
    to_keep = operation(to_keep, hashes[i]);
  }

  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> all_streamlines;
  for (const auto &group : streamlines) {
    all_streamlines.insert(all_streamlines.end(), group.begin(), group.end());
  }

  std::vector<uint32_t> indices;
  indices.reserve(to_keep.size());
  for (const auto &item : to_keep) {
    indices.push_back(item.second);
  }
  std::sort(indices.begin(), indices.end());

  std::vector<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> output;
  output.reserve(indices.size());
  for (uint32_t idx : indices) {
    if (idx < all_streamlines.size()) {
      output.push_back(all_streamlines[idx]);
    }
  }

  return {output, indices};
}
} // namespace trxmmap

#endif
