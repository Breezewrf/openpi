#!/usr/bin/env python3
"""Reset LeRobot MP4 timestamps to zero without re-encoding.

Some dataset conversion pipelines produce MP4 files whose first video frame has
a non-zero presentation timestamp (PTS), while the corresponding LeRobot
parquet timestamps start at zero. LeRobot's timestamp-based decoder rejects
that mismatch with its default 100 microsecond tolerance.

By default this script creates a sibling copy named ``<dataset>_timestamp_fixed``.
The source dataset is never modified.

Examples:

    uv run scripts/fix_lerobot_video_timestamps.py /path/to/dataset --check-only

    uv run scripts/fix_lerobot_video_timestamps.py /path/to/dataset \
        --output-dir /path/to/dataset_openpi
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

DEFAULT_TOLERANCE_S = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Root of the LeRobot dataset to inspect or repair.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for the repaired copy. Defaults to <dataset>_timestamp_fixed.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only inspect timestamps; do not copy or modify any files.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help=f"Maximum accepted absolute start time in seconds (default: {DEFAULT_TOLERANCE_S}).",
    )
    return parser.parse_args()


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required program is not installed or not on PATH: {name}")


def validate_dataset_root(dataset_dir: Path) -> None:
    required = (dataset_dir / "meta" / "info.json", dataset_dir / "data", dataset_dir / "videos")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"Not a LeRobot video dataset; missing: {', '.join(missing)}")

    with (dataset_dir / "meta" / "info.json").open() as f:
        info = json.load(f)
    video_features = [key for key, value in info.get("features", {}).items() if value.get("dtype") == "video"]
    if not video_features:
        raise ValueError(f"Dataset has no video features: {dataset_dir}")


def probe_video(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=start_time,nb_frames,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected exactly one primary video stream in {path}, got {len(streams)}")
    stream = streams[0]
    if "start_time" not in stream:
        raise RuntimeError(f"ffprobe did not report a video start_time for {path}")
    stream["start_time"] = float(stream["start_time"])
    return stream


def find_videos(dataset_dir: Path) -> list[Path]:
    with (dataset_dir / "meta" / "info.json").open() as f:
        info = json.load(f)
    with (dataset_dir / "meta" / "episodes.jsonl").open() as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    video_path_template = info.get("video_path")
    chunks_size = info.get("chunks_size")
    video_keys = sorted(key for key, value in info.get("features", {}).items() if value.get("dtype") == "video")
    if not video_path_template or not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("meta/info.json has an invalid video_path or chunks_size")

    videos = [
        dataset_dir
        / video_path_template.format(
            episode_chunk=episode["episode_index"] // chunks_size,
            episode_index=episode["episode_index"],
            video_key=video_key,
        )
        for episode in episodes
        for video_key in video_keys
    ]
    missing = [path for path in videos if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise ValueError(f"Dataset is missing {len(missing)} metadata-declared videos; first missing: {preview}")

    declared_total = info.get("total_videos")
    if declared_total is not None and declared_total != len(videos):
        raise ValueError(f"meta/info.json declares {declared_total} videos, but metadata resolves to {len(videos)}")

    extras = set((dataset_dir / "videos").rglob("*.mp4")) - set(videos)
    if extras:
        print(f"Warning: ignoring {len(extras)} MP4 files not declared by dataset metadata", file=sys.stderr)
    if not videos:
        raise ValueError(f"No metadata-declared MP4 videos found below {dataset_dir / 'videos'}")
    return videos


def inspect(dataset_dir: Path, tolerance_s: float) -> tuple[list[Path], list[tuple[Path, dict[str, object]]]]:
    videos = find_videos(dataset_dir)
    needs_fix: list[tuple[Path, dict[str, object]]] = []
    starts: dict[float, int] = {}
    for video in videos:
        metadata = probe_video(video)
        start = float(metadata["start_time"])
        starts[start] = starts.get(start, 0) + 1
        if abs(start) > tolerance_s:
            needs_fix.append((video, metadata))

    print(f"Dataset: {dataset_dir}")
    print(f"Videos: {len(videos)}")
    print("Video start-time distribution:")
    for start, count in sorted(starts.items()):
        print(f"  {start:.6f}s: {count}")
    print(f"Outside +/-{tolerance_s:g}s tolerance: {len(needs_fix)}")
    return videos, needs_fix


def remux_video(path: Path, tolerance_s: float) -> None:
    before = probe_video(path)
    temporary = path.with_name(f".{path.stem}.timestamp-fix-{uuid.uuid4().hex}.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-copyts",
                "-start_at_zero",
                "-i",
                str(path),
                "-map",
                "0",
                "-c",
                "copy",
                str(temporary),
            ],
            check=True,
        )
        after = probe_video(temporary)

        if abs(float(after["start_time"])) > tolerance_s:
            raise RuntimeError(
                f"Timestamp repair did not bring {path} within tolerance: start_time={after['start_time']}"
            )
        if before.get("nb_frames") != after.get("nb_frames"):
            raise RuntimeError(
                f"Frame count changed while repairing {path}: {before.get('nb_frames')} -> {after.get('nb_frames')}"
            )
        if (before.get("width"), before.get("height"), before.get("avg_frame_rate")) != (
            after.get("width"),
            after.get("height"),
            after.get("avg_frame_rate"),
        ):
            raise RuntimeError(f"Video geometry or frame rate changed while repairing {path}")

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repair(source: Path, output: Path, tolerance_s: float) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output == source or source in output.parents:
        raise ValueError("Output must be a separate directory outside the source dataset")

    staging = output.with_name(f".{output.name}.fixing-{uuid.uuid4().hex}")
    print(f"Copying dataset to staging directory: {staging}")
    try:
        shutil.copytree(source, staging)
        videos = find_videos(staging)
        repaired = 0
        for index, video in enumerate(videos, start=1):
            if abs(float(probe_video(video)["start_time"])) > tolerance_s:
                remux_video(video, tolerance_s)
                repaired += 1
            if index == 1 or index % 10 == 0 or index == len(videos):
                print(f"Processed {index}/{len(videos)} videos", flush=True)

        _, remaining = inspect(staging, tolerance_s)
        if remaining:
            raise RuntimeError(f"Verification failed: {len(remaining)} videos still have non-zero timestamps")

        staging.rename(output)
        print(f"Repaired videos: {repaired}")
        print(f"Repaired dataset: {output}")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    require_program("ffprobe")
    source = args.dataset_dir.expanduser().resolve()
    validate_dataset_root(source)
    if args.tolerance_s < 0:
        raise ValueError("--tolerance-s must be non-negative")

    _, needs_fix = inspect(source, args.tolerance_s)
    if args.check_only:
        return 0
    if not needs_fix:
        print("All video timestamps are already within tolerance; no repaired copy was created.")
        return 0

    require_program("ffmpeg")
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source.with_name(f"{source.name}_timestamp_fixed")
    )
    repair(source, output, args.tolerance_s)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
