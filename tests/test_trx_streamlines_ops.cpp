#include <gtest/gtest.h>
#include <trx/streamlines_ops.h>

#include <random>

using Streamline = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

namespace {
Streamline make_streamline_ones() { return Streamline::Ones(30, 3); }

Streamline make_streamline_arange() {
  Streamline out(30, 3);
  int idx = 0;
  for (Eigen::Index i = 0; i < out.rows(); ++i) {
    for (Eigen::Index j = 0; j < out.cols(); ++j) {
      out(i, j) = static_cast<float>(idx) + 0.3333f;
      ++idx;
    }
  }
  return out;
}

Streamline random_matrix(std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  Streamline out(30, 3);
  for (Eigen::Index i = 0; i < out.rows(); ++i) {
    for (Eigen::Index j = 0; j < out.cols(); ++j) {
      out(i, j) = dist(rng);
    }
  }
  return out;
}

std::vector<Streamline> make_streamlines_new(const Streamline &base, float noise) {
  std::mt19937 rng(1337);
  std::vector<Streamline> streamlines_new;
  streamlines_new.reserve(5);
  for (int i = 0; i < 5; ++i) {
    if (i < 4) {
      streamlines_new.push_back(base + random_matrix(rng));
    } else {
      streamlines_new.push_back(base + noise * random_matrix(rng));
    }
  }
  return streamlines_new;
}
} // namespace

class StreamlinesOpsIntersectionTest : public ::testing::TestWithParam<std::tuple<int, float, std::vector<uint32_t>>> {
};

// Intersection keeps only streamlines shared between lists at given precision.
TEST_P(StreamlinesOpsIntersectionTest, Intersection) {
  const auto params = GetParam();
  const int precision = std::get<0>(params);
  const float noise = std::get<1>(params);
  const std::vector<uint32_t> expected = std::get<2>(params);
  const Streamline s1 = make_streamline_ones();
  const Streamline s2 = make_streamline_arange();
  const std::vector<Streamline> streamlines_ori = {s1, s2};
  const std::vector<Streamline> streamlines_new = make_streamlines_new(s2, noise);

  auto result = trx::perform_streamlines_operation(trx::intersection, {streamlines_new, streamlines_ori}, precision);
  const auto &indices = result.second;
  EXPECT_EQ(indices, expected);
}

INSTANTIATE_TEST_SUITE_P(StreamlinesOps,
                         StreamlinesOpsIntersectionTest,
                         ::testing::Values(std::make_tuple(0, 0.0001f, std::vector<uint32_t>{4}),
                                           std::make_tuple(1, 0.0001f, std::vector<uint32_t>{4}),
                                           std::make_tuple(2, 0.0001f, std::vector<uint32_t>{4}),
                                           std::make_tuple(4, 0.0001f, std::vector<uint32_t>{}),
                                           std::make_tuple(0, 0.01f, std::vector<uint32_t>{4}),
                                           std::make_tuple(1, 0.01f, std::vector<uint32_t>{4}),
                                           std::make_tuple(2, 0.01f, std::vector<uint32_t>{}),
                                           std::make_tuple(0, 1.0f, std::vector<uint32_t>{})));

class StreamlinesOpsUnionTest : public ::testing::TestWithParam<std::tuple<int, float, size_t>> {};

// Union keeps all streamlines across lists at given precision.
TEST_P(StreamlinesOpsUnionTest, Union) {
  const auto params = GetParam();
  const int precision = std::get<0>(params);
  const float noise = std::get<1>(params);
  const size_t expected = std::get<2>(params);
  const Streamline s1 = make_streamline_ones();
  const Streamline s2 = make_streamline_arange();
  const std::vector<Streamline> streamlines_ori = {s1, s2};
  const std::vector<Streamline> streamlines_new = make_streamlines_new(s2, noise);

  auto result = trx::perform_streamlines_operation(trx::union_maps, {streamlines_new, streamlines_ori}, precision);
  EXPECT_EQ(result.first.size(), expected);
}

INSTANTIATE_TEST_SUITE_P(StreamlinesOps,
                         StreamlinesOpsUnionTest,
                         ::testing::Values(std::make_tuple(0, 0.0001f, static_cast<size_t>(6)),
                                           std::make_tuple(1, 0.0001f, static_cast<size_t>(6)),
                                           std::make_tuple(2, 0.0001f, static_cast<size_t>(6)),
                                           std::make_tuple(4, 0.0001f, static_cast<size_t>(7)),
                                           std::make_tuple(0, 0.01f, static_cast<size_t>(6)),
                                           std::make_tuple(1, 0.01f, static_cast<size_t>(6)),
                                           std::make_tuple(2, 0.01f, static_cast<size_t>(7)),
                                           std::make_tuple(0, 1.0f, static_cast<size_t>(7))));

class StreamlinesOpsDifferenceTest : public ::testing::TestWithParam<std::tuple<int, float, size_t>> {};

// Difference removes streamlines from the second list at given precision.
TEST_P(StreamlinesOpsDifferenceTest, Difference) {
  const auto params = GetParam();
  const int precision = std::get<0>(params);
  const float noise = std::get<1>(params);
  const size_t expected = std::get<2>(params);
  const Streamline s1 = make_streamline_ones();
  const Streamline s2 = make_streamline_arange();
  const std::vector<Streamline> streamlines_ori = {s1, s2};
  const std::vector<Streamline> streamlines_new = make_streamlines_new(s2, noise);

  auto result = trx::perform_streamlines_operation(trx::difference, {streamlines_new, streamlines_ori}, precision);
  EXPECT_EQ(result.first.size(), expected);
}

INSTANTIATE_TEST_SUITE_P(StreamlinesOps,
                         StreamlinesOpsDifferenceTest,
                         ::testing::Values(std::make_tuple(0, 0.0001f, static_cast<size_t>(4)),
                                           std::make_tuple(1, 0.0001f, static_cast<size_t>(4)),
                                           std::make_tuple(2, 0.0001f, static_cast<size_t>(4)),
                                           std::make_tuple(4, 0.0001f, static_cast<size_t>(5)),
                                           std::make_tuple(0, 0.01f, static_cast<size_t>(4)),
                                           std::make_tuple(1, 0.01f, static_cast<size_t>(4)),
                                           std::make_tuple(2, 0.01f, static_cast<size_t>(5)),
                                           std::make_tuple(0, 1.0f, static_cast<size_t>(5))));
