#!/usr/bin/env Rscript
#
# plot_bench.R - Plot trx-cpp benchmark results with ggplot2
#
# Usage:
#   Rscript bench/plot_bench.R [--bench-dir DIR] [--out-dir DIR] [--help]
#
# This script automatically detects benchmark result files in the bench/
# directory and generates plots for:
#   - File sizes (BM_TrxFileSize_Float16)
#   - Translate/write throughput (BM_TrxStream_TranslateWrite)
#   - Query performance (BM_TrxQueryAabb_Slabs)
#
# Expected input files (searched in bench-dir):
#   - results*.json: Main benchmark results (Google Benchmark JSON format)
#   - query_timings.jsonl: Canonical per-query timing distributions (JSONL format)
#   - query_timings_*.jsonl: Legacy/suite-specific timing files (also supported)
#   - rss_samples.jsonl: Memory samples over time (JSONL format, optional)
#

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
})

# Constants
GROUP_LABELS <- c(
  "0" = "no groups",
  "1" = "bundle groups (80)",
  "2" = "connectome groups (1480)"
)

COMPRESSION_LABELS <- c(
  "0" = "store (no zip)",
  "1" = "zip deflate"
)

#' Parse command line arguments
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  bench_dir <- "bench"
  out_dir <- "docs/_static/benchmarks"
  
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--bench-dir") {
      bench_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--out-dir") {
      out_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--help" || args[i] == "-h") {
      cat("Usage: Rscript plot_bench.R [--bench-dir DIR] [--out-dir DIR]\n")
      cat("\n")
      cat("Options:\n")
      cat("  --bench-dir DIR   Directory containing benchmark JSON files (default: bench)\n")
      cat("  --out-dir DIR     Output directory for plots (default: docs/_static/benchmarks)\n")
      cat("  --help, -h        Show this help message\n")
      quit(status = 0)
    } else {
      i <- i + 1
    }
  }
  
  list(bench_dir = bench_dir, out_dir = out_dir)
}

#' Convert benchmark time to seconds
time_to_s <- function(bench) {
  value <- bench$real_time
  unit <- bench$time_unit

  multiplier <- switch(unit,
    "ns" = 1e-9,
    "us" = 1e-6,
    "ms" = 1e-3,
    "s"  = 1,
    1e-9  # default to nanoseconds
  )

  value * multiplier
}

#' Extract base benchmark name
parse_base_name <- function(name) {
  sub("/.*", "", name)
}

#' Load all benchmark result JSON files from a directory
load_benchmarks <- function(bench_dir) {
  json_files <- list.files(bench_dir, pattern = "^results.*\\.json$", full.names = TRUE)
  
  if (length(json_files) == 0) {
    stop("No results*.json files found in ", bench_dir)
  }
  
  cat("Found", length(json_files), "benchmark result file(s):\n")
  for (f in json_files) {
    cat("  -", basename(f), "\n")
  }
  
  all_rows <- list()
  
  for (json_file in json_files) {
    data <- tryCatch({
      fromJSON(json_file, simplifyDataFrame = FALSE)
    }, error = function(e) {
      warning("Failed to parse ", json_file, ": ", e$message)
      return(NULL)
    })
    
    if (is.null(data)) {
      next
    }
    
    benchmarks <- data$benchmarks
    
    if (is.null(benchmarks) || length(benchmarks) == 0) {
      warning("No benchmarks found in ", json_file)
      next
    }
    
    for (bench in benchmarks) {
      name <- bench$name %||% ""
      if (!grepl("^BM_", name)) next
      
      row <- list(
        name = name,
        base = parse_base_name(name),
        real_time_s = time_to_s(bench),
        streamlines = bench$streamlines %||% NA,
        length_profile = bench$length_profile %||% NA,
        compression = bench$compression %||% NA,
        group_case = bench$group_case %||% NA,
        group_count = bench$group_count %||% NA,
        dps = bench$dps %||% NA,
        dpv = bench$dpv %||% NA,
        write_ms = bench$write_ms %||% NA,
        build_ms = bench$build_ms %||% NA,
        file_bytes = bench$file_bytes %||% NA,
        max_rss_kb = bench$max_rss_kb %||% NA,
        query_p50_ms = bench$query_p50_ms %||% NA,
        query_p95_ms = bench$query_p95_ms %||% NA,
        shard_merge_ms = bench$shard_merge_ms %||% NA,
        shard_processes = bench$shard_processes %||% NA,
        source_file = basename(json_file)
      )
      
      all_rows[[length(all_rows) + 1]] <- row
    }
  }
  
  if (length(all_rows) == 0) {
    stop("No valid benchmarks found in any JSON file")
  }
  
  df <- bind_rows(all_rows)
  
  cat("\nLoaded", nrow(df), "benchmark results\n")
  cat("Benchmark types found:\n")
  for (base in unique(df$base)) {
    count <- sum(df$base == base)
    cat("  -", base, ":", count, "results\n")
  }
  
  df
}

#' Estimate file sizes for missing streamline counts via linear extrapolation.
#' Uses a per-(dps, dpv, compression, group_case) linear model fit to measured
#' data to fill in counts not present in the benchmark results (e.g. 10M).
estimate_missing_file_sizes <- function(sub_df) {
  target_sl <- sort(unique(c(sub_df$streamlines, 10000000)))

  combos <- sub_df %>%
    filter(!is.na(file_bytes)) %>%
    group_by(dps, dpv, compression, group_case) %>%
    filter(n() >= 2) %>%
    summarise(.groups = "drop")

  if (nrow(combos) == 0) return(NULL)

  estimated_rows <- list()

  for (i in seq_len(nrow(combos))) {
    d   <- combos$dps[i]
    v   <- combos$dpv[i]
    comp <- combos$compression[i]
    gc  <- combos$group_case[i]

    existing <- sub_df %>%
      filter(dps == d, dpv == v, compression == comp,
             group_case == gc, !is.na(file_bytes))

    missing_sl <- setdiff(target_sl, existing$streamlines)
    if (length(missing_sl) == 0) next

    fit  <- lm(file_bytes ~ streamlines, data = existing)
    pred <- predict(fit, newdata = data.frame(streamlines = as.numeric(missing_sl)))

    template <- existing[1, , drop = FALSE]
    for (j in seq_along(missing_sl)) {
      row            <- template
      row$streamlines <- missing_sl[j]
      row$file_bytes  <- max(0, pred[j])
      row$estimated   <- TRUE
      estimated_rows[[length(estimated_rows) + 1]] <- row
    }
  }

  if (length(estimated_rows) == 0) return(NULL)
  bind_rows(estimated_rows)
}

#' Plot file sizes
plot_file_sizes <- function(df, out_dir) {
  sub_df <- df %>%
    filter(base == "BM_TrxFileSize_Float16") %>%
    filter(!is.na(file_bytes), !is.na(streamlines)) %>%
    mutate(estimated = FALSE)

  if (nrow(sub_df) == 0) {
    cat("No BM_TrxFileSize_Float16 results found, skipping file size plot\n")
    return(invisible(NULL))
  }

  # Extrapolate to missing streamline counts (e.g. 10M not present in benchmark data)
  est_df <- estimate_missing_file_sizes(sub_df)
  if (!is.null(est_df)) {
    cat("Added", nrow(est_df), "estimated file size entries (linear extrapolation)\n")
    sub_df <- bind_rows(sub_df, est_df)
  }

  sub_df <- sub_df %>%
    mutate(
      file_mb = file_bytes / 1e6,
      compression_label = recode(as.character(compression), !!!COMPRESSION_LABELS),
      group_label = recode(
        as.character(ifelse(is.na(group_case), "0", as.character(group_case))),
        !!!GROUP_LABELS),
      dp_label = sprintf("dpv=%d, dps=%d", as.integer(dpv), as.integer(dps)),
      measured = ifelse(estimated, "estimated", "measured"),
      streamlines_f = factor(
        streamlines,
        levels = sort(as.numeric(unique(streamlines))),
        labels = label_number(scale = 1e-6, suffix = "M")(sort(as.numeric(unique(streamlines))))
      )
    )

  n_group_levels <- length(unique(sub_df$group_label))
  plot_height <- if (n_group_levels > 1) 4 + 2 * n_group_levels else 6

  p <- ggplot(sub_df, aes(x = streamlines_f, y = file_mb,
                          fill = dp_label,
                          linetype = compression_label,
                          alpha = measured)) +
    geom_hline(yintercept = 18500, color = "firebrick", linetype = "dashed", linewidth = 0.8) +
    annotate("text", x = Inf, y = 18500, label = "TCK reference 10M (18.5 GB)",
             hjust = 1.05, vjust = -0.4, color = "firebrick", size = 3) +
    geom_col(position = "dodge", color = "grey30", linewidth = 0.5) +
    facet_wrap(~group_label, ncol = 1) +
    scale_y_continuous(labels = label_number()) +
    scale_alpha_manual(
      values = c("measured" = 0.9, "estimated" = 0.45),
      name = ""
    ) +
    labs(
      title = "TRX file size vs streamlines (float16 positions)",
      x = "Streamlines",
      y = "File size (MB)",
      fill = "Data per streamline/vertex",
      linetype = "Compression"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.box = "vertical",
      strip.background = element_rect(fill = "grey90"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  out_path <- file.path(out_dir, "trx_size_vs_streamlines.png")
  ggsave(out_path, p, width = 10, height = plot_height, dpi = 160)
  cat("Saved:", out_path, "\n")
}

#' Plot translate/write performance
plot_translate_write <- function(df, out_dir) {
  sub_df <- df %>%
    filter(base == "BM_TrxStream_TranslateWrite") %>%
    filter(!is.na(real_time_s), !is.na(streamlines))

  if (nrow(sub_df) == 0) {
    cat("No BM_TrxStream_TranslateWrite results found, skipping translate plots\n")
    return(invisible(NULL))
  }

  sub_df <- sub_df %>%
    mutate(
      group_label = recode(as.character(group_case), !!!GROUP_LABELS),
      dp_label = sprintf("dpv=%d, dps=%d", as.integer(dpv), as.integer(dps)),
      rss_gb = max_rss_kb / (1024 * 1024),
      streamlines_f = factor(
        streamlines,
        levels = sort(as.numeric(unique(streamlines))),
        labels = label_number(scale = 1e-6, suffix = "M")(sort(as.numeric(unique(streamlines))))
      )
    )

  # Time plot
  p_time <- ggplot(sub_df, aes(x = streamlines_f, y = real_time_s, fill = dp_label)) +
    geom_col(position = "dodge") +
    facet_wrap(~group_label, ncol = 2) +
    scale_y_continuous(labels = label_number()) +
    labs(
      title = "Translate + stream write throughput",
      x = "Streamlines",
      y = "Time (s)",
      fill = "Data per point"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(fill = "grey90"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  out_path <- file.path(out_dir, "trx_translate_write_time.png")
  ggsave(out_path, p_time, width = 12, height = 5, dpi = 160)
  cat("Saved:", out_path, "\n")

  # RSS plot
  p_rss <- ggplot(sub_df, aes(x = streamlines_f, y = rss_gb, fill = dp_label)) +
    geom_col(position = "dodge") +
    facet_wrap(~group_label, ncol = 2) +
    scale_y_continuous(labels = label_number()) +
    labs(
      title = "Translate + stream write memory usage",
      x = "Streamlines",
      y = "Max RSS (GB)",
      fill = "Data per point"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(fill = "grey90"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  out_path <- file.path(out_dir, "trx_translate_write_rss.png")
  ggsave(out_path, p_rss, width = 12, height = 5, dpi = 160)
  cat("Saved:", out_path, "\n")
}

#' Load query timings from JSONL file
load_query_timings <- function(jsonl_path) {
  if (!file.exists(jsonl_path)) {
    return(NULL)
  }
  
  lines <- readLines(jsonl_path, warn = FALSE)
  lines <- lines[nzchar(lines)]
  
  if (length(lines) == 0) {
    return(NULL)
  }
  
  rows <- lapply(lines, function(line) {
    tryCatch({
      obj <- fromJSON(line, simplifyDataFrame = FALSE)
      list(
        streamlines = obj$streamlines %||% NA,
        group_case = obj$group_case %||% NA,
        group_count = obj$group_count %||% NA,
        dps = obj$dps %||% NA,
        dpv = obj$dpv %||% NA,
        slab_thickness_mm = obj$slab_thickness_mm %||% NA,
        timings_s = I(list(unlist(obj$timings_ms) / 1000))
      )
    }, error = function(e) NULL)
  })
  
  rows <- rows[!sapply(rows, is.null)]
  
  if (length(rows) == 0) {
    return(NULL)
  }
  
  bind_rows(rows)
}

#' Load query timings from canonical and legacy JSONL files
load_all_query_timings <- function(bench_dir) {
  canonical <- file.path(bench_dir, "query_timings.jsonl")
  legacy <- list.files(bench_dir, pattern = "^query_timings.*\\.jsonl$", full.names = TRUE)
  jsonl_paths <- unique(c(canonical, legacy))
  jsonl_paths <- jsonl_paths[file.exists(jsonl_paths)]

  if (length(jsonl_paths) == 0) {
    return(NULL)
  }

  dfs <- lapply(jsonl_paths, function(path) {
    df <- load_query_timings(path)
    if (is.null(df) || nrow(df) == 0) {
      return(NULL)
    }
    df$source_file <- basename(path)
    df
  })
  dfs <- dfs[!sapply(dfs, is.null)]
  if (length(dfs) == 0) {
    return(NULL)
  }
  bind_rows(dfs)
}

#' Plot query timing distributions
plot_query_timings <- function(bench_dir, out_dir) {
  df <- load_all_query_timings(bench_dir)

  if (is.null(df) || nrow(df) == 0) {
    cat("No query_timings*.jsonl found or empty, skipping query timing plot\n")
    return(invisible(NULL))
  }

  # Expand timings into long format, keeping all group/dpv/dps combinations
  timing_data <- df %>%
    mutate(
      streamlines_label = factor(
        label_number(scale = 1e-6, suffix = "M")(streamlines),
        levels = label_number(scale = 1e-6, suffix = "M")(sort(as.numeric(unique(streamlines))))
      ),
      dp_label = sprintf("dpv=%d, dps=%d", as.integer(dpv), as.integer(dps)),
      group_label = recode(as.character(group_case), !!!GROUP_LABELS)
    ) %>%
    select(streamlines, streamlines_label, dp_label, group_label, timings_s) %>%
    unnest(timings_s) %>%
    ungroup()

  n_groups <- length(unique(timing_data$group_label))
  n_dp <- length(unique(timing_data$dp_label))
  plot_height <- max(6, 3 * n_groups)
  plot_width <- max(10, 4 * n_dp)

  p <- ggplot(timing_data, aes(x = streamlines_label, y = timings_s)) +
    geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.size = 0.5) +
    facet_grid(group_label ~ dp_label) +
    labs(
      title = "Slab query timings by group and data profile",
      x = "Streamlines",
      y = "Per-slab query time (s)"
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.background = element_rect(fill = "grey90")
    )

  out_path <- file.path(out_dir, "trx_query_slab_timings.png")
  ggsave(out_path, p, width = plot_width, height = plot_height, dpi = 160)
  cat("Saved:", out_path, "\n")
}

#' Main function
main <- function() {
  args <- parse_args()
  
  # Create output directory
  dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)
  
  cat("\n=== TRX-CPP Benchmark Plotting ===\n\n")
  cat("Benchmark directory:", args$bench_dir, "\n")
  cat("Output directory:", args$out_dir, "\n\n")
  
  # Load benchmark results
  df <- load_benchmarks(args$bench_dir)
  
  cat("\n--- Generating plots ---\n\n")
  
  # Generate plots
  plot_file_sizes(df, args$out_dir)
  plot_translate_write(df, args$out_dir)
  plot_query_timings(args$bench_dir, args$out_dir)
  
  cat("\nDone! Plots saved to:", args$out_dir, "\n")
}

# Define null-coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Run main if executed as script
if (!interactive()) {
  main()
}
