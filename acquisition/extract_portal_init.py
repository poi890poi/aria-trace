"""Extract a portal initialization interval from a recorded session."""

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

import cv2

from .annotations import AnnotationStore
from .session import SessionReader


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _choose_interval(annotations, portal_id: str, route_id: Optional[str]):
    relevant = [
        item for item in annotations
        if item.get("portal_id") == portal_id
        and (route_id is None or item.get("route_id") in (None, route_id))
    ]
    starts = [item for item in relevant if item["kind"] == "world_ready"]
    for start in starts:
        ends = [
            item for item in relevant
            if item["kind"] == "route_start"
            and item["session_time_ns"] > start["session_time_ns"]
        ]
        if ends:
            return start, ends[0]
    raise RuntimeError(
        "No world_ready -> route_start interval for portal {}".format(portal_id)
    )


def extract_initialization(
    session_path: Path,
    output: Path,
    portal_id: str,
    route_id: Optional[str] = None,
    stream_id: Optional[str] = None,
    require_lossless: bool = False,
) -> dict:
    session = SessionReader(session_path)
    annotations = AnnotationStore(session_path).list()
    start, end = _choose_interval(annotations, portal_id, route_id)
    if stream_id is None:
        stream_id = start["stream_id"]
    frames = [
        frame for frame in session.frames_by_stream.get(stream_id, [])
        if start["session_time_ns"] <= frame["session_time_ns"] <= end["session_time_ns"]
    ]
    if not frames:
        raise RuntimeError("The annotated interval contains no frames for {}".format(stream_id))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory is not empty: {}".format(output))
    images_path = output / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    frame_lookup = {frame["frame_index"]: frame for frame in frames}
    extracted = []
    for artifact in session.manifest.get("online_frame_artifacts", []):
        if artifact.get("type") != "OnlineSiftRecorder":
            continue
        database = Path(session_path) / artifact["path"]
        connection = sqlite3.connect(str(database))
        try:
            rows = connection.execute(
                """
                SELECT frame_index, lossless_png, keypoint_count, frame_sha256
                FROM observations
                WHERE stream_id = ? AND session_time_ns BETWEEN ? AND ?
                ORDER BY frame_index
                """,
                (stream_id, start["session_time_ns"], end["session_time_ns"]),
            ).fetchall()
        finally:
            connection.close()
        for frame_index, png, keypoint_count, frame_hash in rows:
            if png is None or frame_index not in frame_lookup:
                continue
            filename = "{:08d}.png".format(frame_index)
            (images_path / filename).write_bytes(png)
            extracted.append(
                {
                    "frame_index": frame_index,
                    "session_time_ns": frame_lookup[frame_index]["session_time_ns"],
                    "image": "images/{}".format(filename),
                    "source_quality": "raw_lossless_evidence",
                    "raw_frame_sha256": frame_hash,
                    "keypoint_count": keypoint_count,
                    "feature_database": artifact["path"],
                }
            )

    if not extracted and require_lossless:
        raise RuntimeError(
            "No lossless online feature frames in the interval. Record with "
            "--online-features sift --feature-lossless-frames."
        )
    if not extracted:
        capture = cv2.VideoCapture(str(session.video_path(stream_id)))
        if not capture.isOpened():
            raise RuntimeError("Cannot open session video for {}".format(stream_id))
        try:
            for frame in frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_index"])
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("Cannot decode frame {}".format(frame["frame_index"]))
                filename = "{:08d}.png".format(frame["frame_index"])
                if not cv2.imwrite(str(images_path / filename), image):
                    raise RuntimeError("Cannot write {}".format(filename))
                extracted.append(
                    {
                        "frame_index": frame["frame_index"],
                        "session_time_ns": frame["session_time_ns"],
                        "image": "images/{}".format(filename),
                        "source_quality": "decoded_compressed_video",
                    }
                )
        finally:
            capture.release()

    manifest = {
        "schema_version": "1.0",
        "source_session": str(Path(session_path).resolve()),
        "source_session_id": session.manifest.get("session_id"),
        "portal_id": portal_id,
        "route_id": route_id,
        "stream_id": stream_id,
        "world_ready_annotation_id": start["annotation_id"],
        "route_start_annotation_id": end["annotation_id"],
        "interval_session_time_ns": [start["session_time_ns"], end["session_time_ns"]],
        "frame_count": len(extracted),
        "source_quality": sorted(set(item["source_quality"] for item in extracted)),
        "frames": extracted,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a portal initialization interval")
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--portal-id", required=True)
    parser.add_argument("--route-id")
    parser.add_argument("--stream")
    parser.add_argument("--require-lossless", action="store_true")
    args = parser.parse_args()
    result = extract_initialization(
        args.session,
        args.output,
        args.portal_id,
        args.route_id,
        args.stream,
        args.require_lossless,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
