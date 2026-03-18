#include <gtest/gtest.h>
#include <trx/trx.h>

#include <map>
#include <string>
#include <vector>

// Helper: split a multi-line string into individual lines.
static std::vector<std::string> split_lines(const std::string &s) {
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = s.find('\n', start);
    if (end == std::string::npos) {
      lines.push_back(s.substr(start));
      break;
    }
    lines.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  return lines;
}

// ── Empty input ───────────────────────────────────────────────────────────────

TEST(FormatGroupsSummary, EmptyGroupsAlwaysReturnsEmptyString) {
  const std::map<std::string, size_t> groups;
  EXPECT_EQ(trx::format_groups_summary(groups), "");
  EXPECT_EQ(trx::format_groups_summary(groups, 0), "");
  EXPECT_EQ(trx::format_groups_summary(groups, -1), "");
  EXPECT_EQ(trx::format_groups_summary(groups, 2), "");
  EXPECT_EQ(trx::format_groups_summary(groups, 2, "  "), "");
}

// ── Flat list (prefix_depth == 0) ─────────────────────────────────────────────

TEST(FormatGroupsSummary, FlatListSingleGroup) {
  const std::map<std::string, size_t> groups = {{"roi_A", 42}};
  EXPECT_EQ(trx::format_groups_summary(groups, 0), "roi_A: 42 streamlines");
}

TEST(FormatGroupsSummary, FlatListMultipleGroups) {
  const std::map<std::string, size_t> groups = {
      {"glasser_Left_V1", 100},
      {"glasser_Left_V2", 200},
      {"glasser_Right_V1", 150},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 0));
  ASSERT_EQ(lines.size(), 3u);
  // std::map iterates in lexicographic order
  EXPECT_EQ(lines[0], "glasser_Left_V1: 100 streamlines");
  EXPECT_EQ(lines[1], "glasser_Left_V2: 200 streamlines");
  EXPECT_EQ(lines[2], "glasser_Right_V1: 150 streamlines");
}

TEST(FormatGroupsSummary, NegativePrefixDepthBehavesLikeFlatList) {
  const std::map<std::string, size_t> groups = {
      {"glasser_Left_V1", 100},
      {"glasser_Left_V2", 200},
      {"glasser_Right_V1", 150},
  };
  EXPECT_EQ(trx::format_groups_summary(groups, -3), trx::format_groups_summary(groups, 0));
}

TEST(FormatGroupsSummary, FlatListNoTrailingNewline) {
  const std::map<std::string, size_t> groups = {{"a", 1}, {"b", 2}};
  const std::string result = trx::format_groups_summary(groups, 0);
  EXPECT_NE(result.back(), '\n');
}

TEST(FormatGroupsSummary, FlatListGroupWithoutUnderscores) {
  const std::map<std::string, size_t> groups = {{"hippocampus", 363}};
  EXPECT_EQ(trx::format_groups_summary(groups, 0), "hippocampus: 363 streamlines");
}

// ── line_prefix ───────────────────────────────────────────────────────────────

TEST(FormatGroupsSummary, LinePrefixAppliedToEveryLineFlat) {
  const std::map<std::string, size_t> groups = {{"a_x", 1}, {"a_y", 2}};
  const auto lines = split_lines(trx::format_groups_summary(groups, 0, "      "));
  for (const auto &line : lines)
    EXPECT_EQ(line.substr(0, 6), "      ") << "Line missing indent: " << line;
}

TEST(FormatGroupsSummary, LinePrefixAppliedToEveryLineCollapsed) {
  const std::map<std::string, size_t> groups = {{"a_x", 1}, {"a_y", 2}};
  const auto lines = split_lines(trx::format_groups_summary(groups, 1, "  "));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0].substr(0, 2), "  ");
}

// ── Collapsing: single group per prefix → shows original name ─────────────────
//
// When only one group maps to a prefix key the original full name is shown,
// not the truncated prefix key. This preserves identity information while still
// benefiting from the collapsed view for groups with many siblings.

TEST(FormatGroupsSummary, SingleGroupPerPrefixShowsOriginalName) {
  // "glasser_Left_V1" → depth-1 key "glasser" (n=1) → full name shown, no wildcard
  // "4S456_RH_A"      → depth-1 key "4S456"   (n=1) → full name shown, no wildcard
  const std::map<std::string, size_t> groups = {
      {"glasser_Left_V1", 100},
      {"4S456_RH_A", 20},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 1));
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "4S456_RH_A: 20 streamlines");
  EXPECT_EQ(lines[1], "glasser_Left_V1: 100 streamlines");
}

TEST(FormatGroupsSummary, SingleGroupPerPrefixNoWildcardAtAnyDepth) {
  const std::map<std::string, size_t> groups = {{"atlas_LH_V1", 99}};
  for (int depth : {0, 1, 2, 3, 10}) {
    const std::string result = trx::format_groups_summary(groups, depth);
    EXPECT_EQ(std::string::npos, result.find('*'))
        << "Unexpected wildcard at depth=" << depth << ": " << result;
  }
}

// ── Collapsing: depth 1 ───────────────────────────────────────────────────────

TEST(FormatGroupsSummary, Depth1CollapsesSamePrefix) {
  const std::map<std::string, size_t> groups = {
      {"glasser_Left_V1", 100},
      {"glasser_Left_V2", 200},
      {"glasser_Right_V1", 150},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 1));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "glasser_*: 450 streamlines (3 groups)");
}

TEST(FormatGroupsSummary, Depth1TwoPrefixes) {
  const std::map<std::string, size_t> groups = {
      {"4S456_LH_A", 10},
      {"4S456_RH_A", 20},
      {"glasser_Left_V1", 100},
      {"glasser_Right_V1", 150},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 1));
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "4S456_*: 30 streamlines (2 groups)");
  EXPECT_EQ(lines[1], "glasser_*: 250 streamlines (2 groups)");
}

// ── Collapsing: depth 2 ───────────────────────────────────────────────────────

TEST(FormatGroupsSummary, Depth2SplitsLeftRight) {
  const std::map<std::string, size_t> groups = {
      {"glasser_Left_V1", 100},
      {"glasser_Left_V2", 200},
      {"glasser_Right_V1", 150},
      {"glasser_Right_V2", 50},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 2));
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "glasser_Left_*: 300 streamlines (2 groups)");
  EXPECT_EQ(lines[1], "glasser_Right_*: 200 streamlines (2 groups)");
}

TEST(FormatGroupsSummary, Depth2RealWorldMix) {
  // Mirrors the example from the original feature request.
  const std::map<std::string, size_t> groups = {
      {"4S456_RH_DorsAttn_Post_5", 28},
      {"4S456_RH_DorsAttn_Post_6", 37},
      {"4S456_RH_DorsAttn_Post_7", 176},
      {"4S456_RH_DorsAttn_PrCv_1", 178},
      {"4S456_RH_Hippocampus", 363},
      {"4S456_RH_Limbic_OFC_1", 70},
      {"4S456_LH_DorsAttn_Post_1", 99},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 2));
  ASSERT_EQ(lines.size(), 2u);
  // LH has only one group → original full name shown, no wildcard
  EXPECT_EQ(lines[0], "4S456_LH_DorsAttn_Post_1: 99 streamlines");
  // RH has six groups → collapsed
  EXPECT_EQ(lines[1], "4S456_RH_*: 852 streamlines (6 groups)");
}

// ── Not enough underscores (fewer delimiters than depth) ─────────────────────

TEST(FormatGroupsSummary, ZeroUnderscoresAtDepth1) {
  // "hippocampus" has 0 underscores; at depth 1 the full name is the key.
  const std::map<std::string, size_t> groups = {{"hippocampus", 363}};
  const auto lines = split_lines(trx::format_groups_summary(groups, 1));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "hippocampus: 363 streamlines");
}

TEST(FormatGroupsSummary, ZeroUnderscoresGroupsListedSeparately) {
  // Names with no underscores each become their own key at any depth.
  const std::map<std::string, size_t> groups = {
      {"amygdala", 100},
      {"hippocampus", 200},
  };
  for (int depth : {1, 2, 5}) {
    const auto lines = split_lines(trx::format_groups_summary(groups, depth));
    ASSERT_EQ(lines.size(), 2u) << "depth=" << depth;
    EXPECT_EQ(lines[0], "amygdala: 100 streamlines") << "depth=" << depth;
    EXPECT_EQ(lines[1], "hippocampus: 200 streamlines") << "depth=" << depth;
  }
}

TEST(FormatGroupsSummary, OneUnderscoreAtDepth2UsesFullName) {
  // "roi_sub" has 1 underscore; at depth 2 there is no second underscore so
  // the full name "roi_sub" becomes the key.
  const std::map<std::string, size_t> groups = {{"roi_sub", 75}};
  const auto lines = split_lines(trx::format_groups_summary(groups, 2));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "roi_sub: 75 streamlines");
}

TEST(FormatGroupsSummary, MixedSufficiencyGroupsHaveDistinctKeys) {
  // "roi" (0 underscores) and "roi_sub" (1 underscore) at depth 2:
  //   "roi"     → key "roi"     (0 delimiters < 2)
  //   "roi_sub" → key "roi_sub" (1 delimiter < 2)
  // Different keys → listed separately.
  const std::map<std::string, size_t> groups = {
      {"roi", 50},
      {"roi_sub", 75},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 2));
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "roi: 50 streamlines");
  EXPECT_EQ(lines[1], "roi_sub: 75 streamlines");
}

TEST(FormatGroupsSummary, ExactTokenCountBoundary) {
  // "a_b" has exactly 1 underscore (2 tokens).
  // At depth 1: key = "a" → both "a_b" and "a_c" collapse under "a".
  // At depth 2: no second '_' → full name is the key → each shown separately.
  const std::map<std::string, size_t> groups = {
      {"a_b", 10},
      {"a_c", 20},
  };
  {
    const auto lines = split_lines(trx::format_groups_summary(groups, 1));
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0], "a_*: 30 streamlines (2 groups)");
  }
  {
    const auto lines = split_lines(trx::format_groups_summary(groups, 2));
    ASSERT_EQ(lines.size(), 2u);
    // n==1 per key → original names shown (which equal the keys in this case)
    EXPECT_EQ(lines[0], "a_b: 10 streamlines");
    EXPECT_EQ(lines[1], "a_c: 20 streamlines");
  }
}

TEST(FormatGroupsSummary, ShortAndLongNamesMixed) {
  // Mix of names with varying underscore counts at depth 3.
  //   "atlas_LH_region_sub1" → key "atlas_LH_region"  (3 tokens, n=2 with sub2)
  //   "atlas_LH_region_sub2" → key "atlas_LH_region"
  //   "atlas_LH"             → key "atlas_LH"          (1 underscore < 3, n=1)
  //   "atlas"                → key "atlas"              (0 underscores,   n=1)
  //   "other_RH_area_part1"  → key "other_RH_area"     (3 tokens,        n=1)
  const std::map<std::string, size_t> groups = {
      {"atlas_LH_region_sub1", 10},
      {"atlas_LH_region_sub2", 20},
      {"atlas_LH", 30},
      {"atlas", 40},
      {"other_RH_area_part1", 5},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 3));
  ASSERT_EQ(lines.size(), 4u);
  EXPECT_EQ(lines[0], "atlas: 40 streamlines");
  EXPECT_EQ(lines[1], "atlas_LH: 30 streamlines");
  EXPECT_EQ(lines[2], "atlas_LH_region_*: 30 streamlines (2 groups)");
  // n==1 → original full name shown, not just "other_RH_area"
  EXPECT_EQ(lines[3], "other_RH_area_part1: 5 streamlines");
}

// ── Depth larger than token count ─────────────────────────────────────────────

TEST(FormatGroupsSummary, DepthExceedingTokenCountEquivalentToFlat) {
  // "a_b_c" has 2 underscores (3 tokens). At depth ≥ 3 there is no 3rd
  // underscore so each full name becomes its own key → output same as flat.
  const std::map<std::string, size_t> groups = {
      {"a_b_c", 10},
      {"a_b_d", 20},
      {"x_y_z", 30},
  };
  const auto lines_flat = split_lines(trx::format_groups_summary(groups, 0));
  const auto lines_d3 = split_lines(trx::format_groups_summary(groups, 3));
  const auto lines_d99 = split_lines(trx::format_groups_summary(groups, 99));
  EXPECT_EQ(lines_d3, lines_flat);
  EXPECT_EQ(lines_d99, lines_flat);
}

// ── Streamline count accuracy ─────────────────────────────────────────────────

TEST(FormatGroupsSummary, CountsSummedCorrectly) {
  const std::map<std::string, size_t> groups = {
      {"net_A_1", 100},
      {"net_A_2", 200},
      {"net_A_3", 300},
      {"net_B_1", 400},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 1));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "net_*: 1000 streamlines (4 groups)");
}

TEST(FormatGroupsSummary, ZeroCountGroupsIncludedInSum) {
  const std::map<std::string, size_t> groups = {
      {"roi_empty", 0},
      {"roi_full", 50},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 1));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "roi_*: 50 streamlines (2 groups)");
}

// ── Output ordering ───────────────────────────────────────────────────────────

TEST(FormatGroupsSummary, FlatOutputIsSortedLexicographically) {
  const std::map<std::string, size_t> groups = {
      {"z_group", 1},
      {"a_group", 2},
      {"m_group", 3},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 0));
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(lines[0], "a_group: 2 streamlines");
  EXPECT_EQ(lines[1], "m_group: 3 streamlines");
  EXPECT_EQ(lines[2], "z_group: 1 streamlines");
}

TEST(FormatGroupsSummary, CollapsedOutputIsSortedLexicographically) {
  const std::map<std::string, size_t> groups = {
      {"z_1", 10}, {"z_2", 10},
      {"a_1", 20}, {"a_2", 20},
  };
  const auto lines = split_lines(trx::format_groups_summary(groups, 1));
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "a_*: 40 streamlines (2 groups)");
  EXPECT_EQ(lines[1], "z_*: 20 streamlines (2 groups)");
}

// ── No trailing newline ───────────────────────────────────────────────────────

TEST(FormatGroupsSummary, CollapsedOutputNoTrailingNewline) {
  const std::map<std::string, size_t> groups = {{"a_x", 1}, {"a_y", 2}, {"b_x", 3}};
  const std::string result = trx::format_groups_summary(groups, 1);
  EXPECT_NE(result.back(), '\n');
}
