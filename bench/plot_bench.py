import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LENGTH_LABELS = {
    0: "mixed",
    1: "short (20-120mm)",
    2: "medium (80-260mm)",
    3: "long (200-500mm)",
}
GROUP_LABELS = {
    0: "no groups",
    1: "bundle groups (80)",
    2: "connectome groups (4950)",
}
COMPRESSION_LABELS = {0: "store (no zip)", 1: "zip deflate"}


def _parse_base_name(name: str) -> str:
    return name.split("/")[0]


def _time_to_ms(bench: dict) -> float:
    value = bench.get("real_time", 0.0)
    unit = bench.get("time_unit", "ns")
    if unit == "ns":
        return value / 1e6
    if unit == "us":
        return value / 1e3
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1e3
    return value / 1e6


def load_benchmarks(path: Path) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f)

    rows = []
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")
        if not name.startswith("BM_"):
            continue
        rows.append(
            {
                "name": name,
                "base": _parse_base_name(name),
                "real_time_ms": _time_to_ms(bench),
                "streamlines": bench.get("streamlines"),
                "length_profile": bench.get("length_profile"),
                "compression": bench.get("compression"),
                "group_case": bench.get("group_case"),
                "group_count": bench.get("group_count"),
                "dps": bench.get("dps"),
                "dpv": bench.get("dpv"),
                "write_ms": bench.get("write_ms"),
                "file_bytes": bench.get("file_bytes"),
                "max_rss_kb": bench.get("max_rss_kb"),
                "query_p50_ms": bench.get("query_p50_ms"),
                "query_p95_ms": bench.get("query_p95_ms"),
            }
        )

    return pd.DataFrame(rows)


def plot_file_sizes(df: pd.DataFrame, output_dir: Path) -> None:
    sub = df[df["base"] == "BM_TrxFileSize_Float16"].copy()
    if sub.empty:
        return
    sub["file_mb"] = sub["file_bytes"] / 1e6
    sub["length_label"] = sub["length_profile"].map(LENGTH_LABELS)
    sub["dp_label"] = "dpv=" + sub["dpv"].astype(int).astype(str) + ", dps=" + sub["dps"].astype(int).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for compression, ax in zip([0, 1], axes):
        scomp = sub[sub["compression"] == compression]
        for (length_label, dp_label), series in scomp.groupby(["length_label", "dp_label"]):
            series = series.sort_values("streamlines")
            ax.plot(
                series["streamlines"],
                series["file_mb"],
                marker="o",
                label=f"{length_label}, {dp_label}",
            )
        ax.set_title(COMPRESSION_LABELS.get(compression, str(compression)))
        ax.set_xlabel("streamlines")
        ax.grid(True)
        ax.legend(loc="best", fontsize="x-small")

    axes[0].set_ylabel("file size (MB)")
    fig.suptitle("TRX file size vs streamlines (float16 positions)")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "trx_size_vs_streamlines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_translate_series(df: pd.DataFrame, output_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    sub = df[df["base"] == "BM_TrxStream_TranslateWrite"].copy()
    if sub.empty:
        return
    sub["group_label"] = sub["group_case"].map(GROUP_LABELS)
    sub["dp_label"] = "dpv=" + sub["dpv"].astype(int).astype(str) + ", dps=" + sub["dps"].astype(int).astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (group_label, gsub) in zip(axes, sub.groupby("group_label")):
        for dp_label, series in gsub.groupby("dp_label"):
            series = series.sort_values("streamlines")
            ax.plot(series["streamlines"], series[metric], marker="o", label=dp_label)
        ax.set_title(group_label)
        ax.set_xlabel("streamlines")
        ax.grid(True)
        ax.legend(loc="best", fontsize="x-small")
    axes[0].set_ylabel(ylabel)
    fig.suptitle("Translate + stream write throughput")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_translate_write(df: pd.DataFrame, output_dir: Path) -> None:
    sub = df[df["base"] == "BM_TrxStream_TranslateWrite"].copy()
    if sub.empty:
        return
    sub["rss_mb"] = sub["max_rss_kb"] / 1024.0
    _plot_translate_series(
        sub,
        output_dir,
        metric="real_time_ms",
        ylabel="time (ms)",
        filename="trx_translate_write_time.png",
    )
    _plot_translate_series(
        sub,
        output_dir,
        metric="rss_mb",
        ylabel="max RSS (MB)",
        filename="trx_translate_write_rss.png",
    )


def load_query_timings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_query_timings(path: Path, output_dir: Path, group_case: int, dpv: int, dps: int) -> None:
    rows = load_query_timings(path)
    if not rows:
        return
    rows = [
        r
        for r in rows
        if r.get("group_case") == group_case and r.get("dpv") == dpv and r.get("dps") == dps
    ]
    if not rows:
        return
    rows.sort(key=lambda r: r["streamlines"])
    data = [r["timings_ms"] for r in rows]
    labels = [str(r["streamlines"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(
        f"Slab query timings ({GROUP_LABELS.get(group_case, group_case)}, dpv={dpv}, dps={dps})"
    )
    ax.set_xlabel("streamlines")
    ax.set_ylabel("per-slab query time (ms)")
    ax.grid(True, axis="y")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "trx_query_slab_timings.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trx-cpp benchmark results.")
    parser.add_argument("bench_json", type=Path, help="Path to benchmark JSON output.")
    parser.add_argument("--query-json", type=Path, help="Path to slab timing JSONL file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/_static/benchmarks"),
        help="Directory to save PNGs.",
    )
    parser.add_argument("--group-case", type=int, default=0, help="Group case filter for query plot.")
    parser.add_argument("--dpv", type=int, default=0, help="DPV filter for query plot.")
    parser.add_argument("--dps", type=int, default=0, help="DPS filter for query plot.")
    args = parser.parse_args()

    df = load_benchmarks(args.bench_json)
    if df.empty:
        raise SystemExit("No benchmarks found in JSON file.")

    plot_file_sizes(df, args.out_dir)
    plot_translate_write(df, args.out_dir)
    if args.query_json:
        plot_query_timings(args.query_json, args.out_dir, args.group_case, args.dpv, args.dps)


if __name__ == "__main__":
    main()
