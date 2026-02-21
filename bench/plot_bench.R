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

#' Convert benchmark time to milliseconds
time_to_ms <- function(bench) {
  value <- bench$real_time
  unit <- bench$time_unit
  
  multiplier <- switch(unit,
    "ns" = 1e-6,
    "us" = 1e-3,
    "ms" = 1,
    "s" = 1e3,
    1e-6  # default to nanoseconds
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
        real_time_ms = time_to_ms(bench),
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

#' Plot file sizes
plot_file_sizes <- function(df, out_dir) {
  sub_df <- df %>%
    filter(base == "BM_TrxFileSize_Float16") %>%
    filter(!is.na(file_bytes), !is.na(streamlines))
  
  if (nrow(sub_df) == 0) {
    cat("No BM_TrxFileSize_Float16 results found, skipping file size plot\n")
    return(invisible(NULL))
  }
  
  sub_df <- sub_df %>%
    mutate(
      file_mb = file_bytes / 1e6,
      compression_label = recode(as.character(compression), !!!COMPRESSION_LABELS),
      group_label = recode(
        as.character(ifelse(is.na(group_case), "0", as.character(group_case))),
        !!!GROUP_LABELS),
      dp_label = sprintf("dpv=%d, dps=%d", as.integer(dpv), as.integer(dps))
    )

  n_group_levels <- length(unique(sub_df$group_label))
  plot_height <- if (n_group_levels > 1) 5 + 3 * n_group_levels else 7

  p <- ggplot(sub_df, aes(x = streamlines, y = file_mb,
                          color = dp_label)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    facet_grid(group_label ~ compression_label) +
    scale_x_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
    scale_y_continuous(labels = label_number()) +
    labs(
      title = "TRX file size vs streamlines (float16 positions)",
      x = "Streamlines",
      y = "File size (MB)",
      color = "Data per streamline/vertex"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      legend.box = "vertical",
      strip.background = element_rect(fill = "grey90")
    )

  out_path <- file.path(out_dir, "trx_size_vs_streamlines.png")
  ggsave(out_path, p, width = 12, height = plot_height, dpi = 160)
  cat("Saved:", out_path, "\n")
}

#' Plot translate/write performance
plot_translate_write <- function(df, out_dir) {
  sub_df <- df %>%
    filter(base == "BM_TrxStream_TranslateWrite") %>%
    filter(!is.na(real_time_ms), !is.na(streamlines))
  
  if (nrow(sub_df) == 0) {
    cat("No BM_TrxStream_TranslateWrite results found, skipping translate plots\n")
    return(invisible(NULL))
  }
  
  sub_df <- sub_df %>%
    mutate(
      group_label = recode(as.character(group_case), !!!GROUP_LABELS),
      dp_label = sprintf("dpv=%d, dps=%d", as.integer(dpv), as.integer(dps)),
      rss_mb = max_rss_kb / 1024
    )
  
  # Time plot
  p_time <- ggplot(sub_df, aes(x = streamlines, y = real_time_ms, 
                                color = dp_label)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    facet_wrap(~group_label, ncol = 2) +
    scale_x_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
    scale_y_continuous(labels = label_number()) +
    labs(
      title = "Translate + stream write throughput",
      x = "Streamlines",
      y = "Time (ms)",
      color = "Data per point"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(fill = "grey90")
    )
  
  out_path <- file.path(out_dir, "trx_translate_write_time.png")
  ggsave(out_path, p_time, width = 12, height = 5, dpi = 160)
  cat("Saved:", out_path, "\n")
  
  # RSS plot
  p_rss <- ggplot(sub_df, aes(x = streamlines, y = rss_mb, 
                               color = dp_label)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    facet_wrap(~group_label, ncol = 2) +
    scale_x_continuous(labels = label_number(scale = 1e-6, suffix = "M")) +
    scale_y_continuous(labels = label_number()) +
    labs(
      title = "Translate + stream write memory usage",
      x = "Streamlines",
      y = "Max RSS (MB)",
      color = "Data per point"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(fill = "grey90")
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
        timings_ms = I(list(unlist(obj$timings_ms)))
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
plot_query_timings <- function(bench_dir, out_dir, group_case = 0, dpv = 0, dps = 0) {
  df <- load_all_query_timings(bench_dir)
  
  if (is.null(df) || nrow(df) == 0) {
    cat("No query_timings*.jsonl found or empty, skipping query timing plot\n")
    return(invisible(NULL))
  }
  
  # Filter by specified conditions
  df_filtered <- df %>%
    filter(
      group_case == !!group_case,
      dpv == !!dpv,
      dps == !!dps
    )
  
  if (nrow(df_filtered) == 0) {
    cat("No query timings matching filters (group_case=", group_case, 
        ", dpv=", dpv, ", dps=", dps, "), skipping plot\n", sep = "")
    return(invisible(NULL))
  }
  
  # Expand timings into long format
  timing_data <- df_filtered %>%
    mutate(streamlines_label = format(streamlines, big.mark = ",")) %>%
    select(streamlines, streamlines_label, timings_ms) %>%
    unnest(timings_ms) %>%
    group_by(streamlines, streamlines_label) %>%
    mutate(query_id = row_number()) %>%
    ungroup()
  
  # Create boxplot
  group_label <- GROUP_LABELS[as.character(group_case)]
  
  p <- ggplot(timing_data, aes(x = streamlines_label, y = timings_ms)) +
    geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.size = 0.5) +
    labs(
      title = sprintf("Slab query timings (%s, dpv=%d, dps=%d)", 
                      group_label, dpv, dps),
      x = "Streamlines",
      y = "Per-slab query time (ms)"
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  out_path <- file.path(out_dir, "trx_query_slab_timings.png")
  ggsave(out_path, p, width = 10, height = 6, dpi = 160)
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
  plot_query_timings(args$bench_dir, args$out_dir, group_case = 0, dpv = 0, dps = 0)
  
  cat("\nDone! Plots saved to:", args$out_dir, "\n")
}

# Define null-coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Run main if executed as script
if (!interactive()) {
  main()
}
