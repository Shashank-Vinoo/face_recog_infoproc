#!/usr/bin/env python3
"""
Face verification/identification for single-face or multi-face images.

Encoding supports three modes:
- auto: detect a face first, then fallback to full-image bounds
- detect: require face detection
- full: force encoding over the entire image bounds
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Suppress a third-party deprecation warning emitted by face_recognition_models.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\.",
    category=UserWarning,
    module=r"face_recognition_models(\..*)?$",
)

try:
    import face_recognition
except ImportError as exc:
    face_recognition = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


NO_MATCH_LABEL = "NO_MATCH"
CACHE_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_KNOWN_DIR = Path(__file__).resolve().parent / "known_faces"
DEFAULT_TEST_INPUT_DIR = Path(__file__).resolve().parent / "test_inputs"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent / "already_checked_images"
ENCODING_MODES = {"auto", "detect", "full"}

LOGGER = logging.getLogger("match_face")


def _ensure_face_recognition_available() -> None:
    """Raise a clear error if face_recognition is not installed."""
    if face_recognition is None:
        raise RuntimeError(
            "face_recognition is required but not installed. "
            "Install with: pip install face_recognition"
        ) from _IMPORT_ERROR


def _is_supported_image(path: Path) -> bool:
    """Return True if path has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _file_signature(path: Path) -> Tuple[int, int]:
    """Return metadata tuple used for cache invalidation: (mtime_ns, size)."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _load_image_from_source(
    image_source: Union[str, Path, np.ndarray, bytes]
) -> Optional[np.ndarray]:
    """
    Load image from a file path, raw bytes, or ndarray.

    Returns an RGB ndarray expected by face_recognition, or None on failure.
    """
    _ensure_face_recognition_available()

    try:
        if isinstance(image_source, np.ndarray):
            image = image_source
        elif isinstance(image_source, (str, Path)):
            image = face_recognition.load_image_file(str(image_source))
        elif isinstance(image_source, bytes):
            image = face_recognition.load_image_file(io.BytesIO(image_source))
        else:
            LOGGER.error("Unsupported image source type: %s", type(image_source))
            return None
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to read image '%s': %s", image_source, exc)
        return None

    if image is None:
        return None

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        LOGGER.warning("Unsupported image shape: %s", getattr(image, "shape", None))
        return None

    return image


def encode_face(
    image_path: Union[str, Path, np.ndarray, bytes],
    mode: str = "auto",
) -> Optional[np.ndarray]:
    """
    Compute one 128-D face embedding from a single-face image.

    Args:
        image_path: Path, bytes, or ndarray representing a face image.
        mode: Encoding mode: auto, detect, or full.

    Returns:
        128-D float32 numpy array if encoding succeeds, otherwise None.
    """
    all_faces = encode_faces(image_path, mode=mode)
    if not all_faces:
        return None

    # For single-face API compatibility, pick the largest detected face.
    selected = max(
        all_faces,
        key=lambda item: max(1, (item["location"][2] - item["location"][0]) * (item["location"][1] - item["location"][3])),
    )
    return selected["encoding"]


def encode_faces(
    image_path: Union[str, Path, np.ndarray, bytes],
    mode: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Compute 128-D face embeddings for all faces in an image.

    Args:
        image_path: Path, bytes, or ndarray representing a face image.
        mode: Encoding mode: auto, detect, or full.

    Returns:
        List of dicts with keys:
        - location: (top, right, bottom, left)
        - encoding: 128-D float32 numpy array
    """
    image = _load_image_from_source(image_path)
    if image is None:
        return []

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        LOGGER.warning("Image has invalid size: (%s, %s)", height, width)
        return []

    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in ENCODING_MODES:
        LOGGER.warning("Unknown encode mode '%s'. Falling back to 'auto'.", mode)
        mode_normalized = "auto"

    def _encode_from_detection() -> List[Dict[str, Any]]:
        try:
            locations = face_recognition.face_locations(image, model="hog")
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Face detection failed for image '%s': %s", image_path, exc)
            return []

        if not locations:
            LOGGER.warning("No face detected in image '%s'.", image_path)
            return []

        try:
            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=locations,
                num_jitters=1,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Encoding with detected face failed for image '%s': %s", image_path, exc)
            return []

        if not encodings:
            LOGGER.warning("No encoding produced for detected face in '%s'.", image_path)
            return []

        if len(encodings) != len(locations):
            LOGGER.warning(
                "Detected %s faces but produced %s encodings for '%s'.",
                len(locations),
                len(encodings),
                image_path,
            )

        results: List[Dict[str, Any]] = []
        for location, raw_encoding in zip(locations, encodings):
            encoding = np.asarray(raw_encoding, dtype=np.float32)
            if encoding.shape != (128,):
                LOGGER.warning("Unexpected encoding shape for '%s': %s", image_path, encoding.shape)
                continue
            results.append({"location": tuple(location), "encoding": encoding})
        return results

    def _encode_from_full_bounds() -> List[Dict[str, Any]]:
        full_bounds = [(0, width, height, 0)]  # top, right, bottom, left
        try:
            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=full_bounds,
                num_jitters=1,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Encoding failed for image '%s': %s", image_path, exc)
            return []

        if not encodings:
            LOGGER.warning("No encoding produced for image '%s'.", image_path)
            return []

        encoding = np.asarray(encodings[0], dtype=np.float32)
        if encoding.shape != (128,):
            LOGGER.warning("Unexpected encoding shape for '%s': %s", image_path, encoding.shape)
            return []
        return [{"location": full_bounds[0], "encoding": encoding}]

    if mode_normalized == "detect":
        return _encode_from_detection()
    if mode_normalized == "full":
        return _encode_from_full_bounds()

    detected_encodings = _encode_from_detection()
    if detected_encodings:
        return detected_encodings

    return _encode_from_full_bounds()


def _safe_load_cache(cache_path: Path) -> Dict[str, Any]:
    """Load cache if possible; otherwise return an empty cache structure."""
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to load cache '%s': %s. Rebuilding.", cache_path, exc)
        return {}

    if not isinstance(data, dict):
        LOGGER.warning("Invalid cache format in '%s'. Rebuilding.", cache_path)
        return {}

    return data


def _iter_known_images(known_dir: Path) -> Iterable[Path]:
    """Yield supported image files in deterministic order."""
    return sorted(
        (p for p in known_dir.iterdir() if p.is_file() and _is_supported_image(p)),
        key=lambda p: p.name.lower(),
    )


def _iter_input_images(input_dir: Path) -> Iterable[Path]:
    """Yield supported input image files in deterministic order."""
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and _is_supported_image(p)),
        key=lambda p: p.name.lower(),
    )


def _build_unique_destination_path(destination_dir: Path, file_name: str) -> Path:
    """Return a non-conflicting destination path in destination_dir."""
    destination = destination_dir / file_name
    if not destination.exists():
        return destination

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    timestamp_ms = int(time.time() * 1000)
    sequence = 1
    while True:
        candidate = destination_dir / f"{stem}_{timestamp_ms}_{sequence}{suffix}"
        if not candidate.exists():
            return candidate
        sequence += 1


def load_or_build_db(
    known_dir: Union[str, Path],
    cache_path: Union[str, Path] = "encodings_cache.pkl",
    encode_mode: str = "auto",
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Load known encodings from cache or build/refresh them from image files.

    Cache invalidation checks per-file metadata (mtime_ns + size). Changed/new
    files are re-encoded; removed files are dropped.

    Args:
        known_dir: Directory containing known face crops.
        cache_path: Path to pickle cache file.
        encode_mode: Encoding mode used for known faces: auto, detect, or full.

    Returns:
        Tuple of:
        - encodings_np: shape (N, 128), dtype float32
        - labels_list: list of N labels (filename stem)
        - meta_dict: details including per-file metadata and cache status
    """
    _ensure_face_recognition_available()

    known_dir_path = Path(known_dir).resolve()
    cache_file = Path(cache_path).resolve()
    encode_mode_normalized = str(encode_mode).strip().lower()
    if encode_mode_normalized not in ENCODING_MODES:
        raise ValueError(
            f"Invalid encode_mode '{encode_mode}'. Expected one of: {sorted(ENCODING_MODES)}"
        )

    if not known_dir_path.exists():
        raise FileNotFoundError(f"Known directory does not exist: {known_dir_path}")
    if not known_dir_path.is_dir():
        raise NotADirectoryError(f"Known path is not a directory: {known_dir_path}")

    cached = _safe_load_cache(cache_file)
    cached_records = {}
    cache_dir_matches = (
        cached.get("version") == CACHE_VERSION
        and cached.get("known_dir") == str(known_dir_path)
        and cached.get("encoding_mode") == encode_mode_normalized
        and isinstance(cached.get("records"), dict)
    )
    if cache_dir_matches:
        cached_records = cached["records"]

    current_files = list(_iter_known_images(known_dir_path))
    current_file_keys = {str(path.resolve()) for path in current_files}
    records: Dict[str, Dict[str, Any]] = {}
    reused_count = 0
    rebuilt_count = 0
    skipped_count = 0

    for img_path in current_files:
        file_key = str(img_path.resolve())
        try:
            mtime_ns, size = _file_signature(img_path)
        except OSError as exc:
            LOGGER.warning("Cannot stat '%s': %s. Skipping.", img_path, exc)
            skipped_count += 1
            continue

        label = img_path.stem
        previous = cached_records.get(file_key)
        can_reuse = (
            isinstance(previous, dict)
            and previous.get("mtime_ns") == mtime_ns
            and previous.get("size") == size
            and previous.get("label") == label
            and isinstance(previous.get("encoding"), np.ndarray)
            and previous["encoding"].shape == (128,)
        )

        if can_reuse:
            records[file_key] = previous
            reused_count += 1
            continue

        encoding = encode_face(img_path, mode=encode_mode_normalized)
        if encoding is None:
            skipped_count += 1
            continue

        records[file_key] = {
            "path": file_key,
            "label": label,
            "mtime_ns": mtime_ns,
            "size": size,
            "encoding": encoding.astype(np.float32),
        }
        rebuilt_count += 1

    removed_files = sorted(set(cached_records.keys()) - current_file_keys)

    ordered_records = sorted(records.values(), key=lambda item: Path(item["path"]).name.lower())
    labels = [record["label"] for record in ordered_records]
    if ordered_records:
        encodings_np = np.vstack([record["encoding"] for record in ordered_records]).astype(np.float32)
    else:
        encodings_np = np.empty((0, 128), dtype=np.float32)

    meta = {
        "known_dir": str(known_dir_path),
        "encoding_mode": encode_mode_normalized,
        "cache_path": str(cache_file),
        "num_known_files": len(current_files),
        "num_valid_encodings": int(encodings_np.shape[0]),
        "reused_from_cache": reused_count,
        "rebuilt_encodings": rebuilt_count,
        "skipped_files": skipped_count,
        "removed_files": removed_files,
        "files": {
            record["path"]: {
                "label": record["label"],
                "mtime_ns": record["mtime_ns"],
                "size": record["size"],
            }
            for record in ordered_records
        },
    }

    save_payload = {
        "version": CACHE_VERSION,
        "known_dir": str(known_dir_path),
        "encoding_mode": encode_mode_normalized,
        "records": {record["path"]: record for record in ordered_records},
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as handle:
            pickle.dump(save_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to save cache '%s': %s", cache_file, exc)
        meta["cache_saved"] = False
    else:
        meta["cache_saved"] = True

    return encodings_np, labels, meta


def match_face(
    input_encoding: Optional[np.ndarray],
    db_encodings: np.ndarray,
    db_labels: Sequence[str],
    threshold: float = 0.6,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Match one input encoding against known encodings using Euclidean distance.

    Args:
        input_encoding: 128-D embedding for the query face.
        db_encodings: shape (N, 128) database embeddings.
        db_labels: labels corresponding to db_encodings rows.
        threshold: max distance to consider as a match.
        top_k: number of nearest candidates to include in debug output.

    Returns:
        Dict with fields: match_label, match_distance, threshold, top_k, status.
    """
    result: Dict[str, Any] = {
        "match_label": NO_MATCH_LABEL,
        "match_distance": None,
        "threshold": float(threshold),
        "top_k": [],
        "status": NO_MATCH_LABEL,
    }

    if input_encoding is None or input_encoding.shape != (128,):
        return result

    if db_encodings.size == 0 or len(db_labels) == 0:
        return result

    if db_encodings.ndim != 2 or db_encodings.shape[1] != 128:
        LOGGER.error("Invalid db_encodings shape: %s", db_encodings.shape)
        return result

    if len(db_labels) != db_encodings.shape[0]:
        LOGGER.error(
            "Mismatched db sizes: %s encodings vs %s labels",
            db_encodings.shape[0],
            len(db_labels),
        )
        return result

    distances = np.linalg.norm(db_encodings - input_encoding.reshape(1, 128), axis=1)
    ranked_indices = np.argsort(distances)
    k = max(0, min(int(top_k), len(ranked_indices)))

    top_candidates: List[Dict[str, Any]] = []
    for idx in ranked_indices[:k]:
        top_candidates.append(
            {
                "label": str(db_labels[idx]),
                "distance": float(distances[idx]),
            }
        )

    if len(ranked_indices) == 0:
        result["top_k"] = top_candidates
        return result

    best_idx = int(ranked_indices[0])
    best_distance = float(distances[best_idx])
    best_label = str(db_labels[best_idx])

    result["match_distance"] = best_distance
    result["top_k"] = top_candidates

    if best_distance <= float(threshold):
        result["match_label"] = best_label
        result["status"] = "MATCH"

    return result


def match_faces(
    input_faces: Sequence[Dict[str, Any]],
    db_encodings: np.ndarray,
    db_labels: Sequence[str],
    threshold: float = 0.6,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Match all detected faces in one image and return attendance-friendly output.

    Args:
        input_faces: sequence of dicts containing location + encoding.
        db_encodings: shape (N, 128) database embeddings.
        db_labels: labels corresponding to db_encodings rows.
        threshold: max distance to consider as a match.
        top_k: number of nearest candidates per detected face.

    Returns:
        Dict with per-face match info and aggregated presence list.
    """
    faces_output: List[Dict[str, Any]] = []
    present_labels: List[str] = []
    seen_labels: set[str] = set()

    for idx, face_item in enumerate(input_faces):
        encoding = face_item.get("encoding")
        location = face_item.get("location")
        single = match_face(encoding, db_encodings, db_labels, threshold=threshold, top_k=top_k)
        is_present = single.get("status") == "MATCH"
        label = str(single.get("match_label", NO_MATCH_LABEL))

        if is_present and label != NO_MATCH_LABEL and label not in seen_labels:
            seen_labels.add(label)
            present_labels.append(label)

        faces_output.append(
            {
                "face_index": idx,
                "location": list(location) if location is not None else None,
                "match_label": label,
                "match_distance": single.get("match_distance"),
                "status": single.get("status", NO_MATCH_LABEL),
                "present": bool(is_present),
                "top_k": single.get("top_k", []),
            }
        )

    return {
        "status": "PRESENT" if present_labels else NO_MATCH_LABEL,
        "threshold": float(threshold),
        "num_faces_detected": len(input_faces),
        "num_present": len(present_labels),
        "present_labels": present_labels,
        "faces": faces_output,
    }


def _run_match_for_input(
    image_path: Union[str, Path],
    db_encodings: np.ndarray,
    db_labels: Sequence[str],
    threshold: float,
    top_k: int,
    encoding_mode: str,
    multi_face: bool,
) -> Dict[str, Any]:
    """Run matching for one input image using single-face or multi-face mode."""
    if multi_face:
        query_faces = encode_faces(image_path, mode=encoding_mode)
        return match_faces(query_faces, db_encodings, db_labels, threshold, top_k)

    query_encoding = encode_face(image_path, mode=encoding_mode)
    return match_face(query_encoding, db_encodings, db_labels, threshold, top_k)


def _format_simple_output(output: Dict[str, Any], multi_face: bool) -> str:
    """Format result for default non-JSON CLI output."""
    if multi_face:
        present_labels = output.get("present_labels", [])
        if present_labels:
            return ",".join(str(label) for label in present_labels)
        return NO_MATCH_LABEL

    if output.get("status") == "MATCH":
        return str(output.get("match_label", NO_MATCH_LABEL))
    return NO_MATCH_LABEL


def _watch_loop(args: argparse.Namespace) -> int:
    """Continuously process new files from input dir and move to processed dir."""
    input_dir = Path(args.watch_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    poll_interval = max(0.1, float(args.poll_interval))
    min_file_age = max(0.0, float(args.min_file_age))
    max_cycles = max(0, int(args.max_cycles))
    cycle_count = 0

    LOGGER.info("Watching input directory: %s", input_dir)
    LOGGER.info("Processed files directory: %s", processed_dir)

    try:
        while True:
            cycle_count += 1
            now = time.time()
            candidates: List[Path] = []
            for image_path in _iter_input_images(input_dir):
                try:
                    age_seconds = now - image_path.stat().st_mtime
                except OSError:
                    continue
                if age_seconds >= min_file_age:
                    candidates.append(image_path)

            if candidates:
                db_encodings, db_labels, _meta = load_or_build_db(
                    args.known_dir,
                    args.cache_path,
                    encode_mode=args.encoding_mode,
                )

                for image_path in candidates:
                    result: Dict[str, Any]
                    try:
                        result = _run_match_for_input(
                            image_path,
                            db_encodings,
                            db_labels,
                            args.threshold,
                            args.top_k,
                            args.encoding_mode,
                            args.multi_face,
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        result = {
                            "status": "ERROR",
                            "error": str(exc),
                            "input_file": str(image_path),
                        }
                    finally:
                        destination = _build_unique_destination_path(processed_dir, image_path.name)
                        try:
                            shutil.move(str(image_path), str(destination))
                        except Exception as move_exc:  # pylint: disable=broad-except
                            LOGGER.error(
                                "Failed to move processed file '%s' to '%s': %s",
                                image_path,
                                destination,
                                move_exc,
                            )

                    if args.json:
                        print(
                            json.dumps(
                                {
                                    "input_file": str(image_path),
                                    "processed_file": str(destination),
                                    "result": result,
                                },
                                ensure_ascii=True,
                            )
                        )
                    else:
                        if result.get("status") == "ERROR":
                            print(f"{image_path.name}: ERROR")
                        else:
                            print(f"{image_path.name}: {_format_simple_output(result, args.multi_face)}")

            if max_cycles and cycle_count >= max_cycles:
                return 0

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        return 0


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Match face image(s) against a known face directory. "
            "Supports one-off matching and continuous watch mode."
        )
    )
    parser.add_argument(
        "--known_dir",
        default=str(DEFAULT_KNOWN_DIR),
        help=(
            "Directory of known cropped face images. "
            f"Default: {DEFAULT_KNOWN_DIR}"
        ),
    )
    parser.add_argument(
        "--input",
        help="Input image path for one-off matching. Required unless --watch is used.",
    )
    parser.add_argument(
        "--cache_path",
        default="encodings_cache.pkl",
        help="Path to cache file (pickle). Default: encodings_cache.pkl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Distance threshold for a positive match. Default: 0.6",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of nearest candidates to include. Default: 5",
    )
    parser.add_argument(
        "--encoding_mode",
        default="auto",
        choices=sorted(ENCODING_MODES),
        help="Face encoding mode for known and input images. Default: auto",
    )
    parser.add_argument(
        "--multi_face",
        action="store_true",
        help="Detect and match all faces in the input image.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch an input folder, process new images, and move them after checking.",
    )
    parser.add_argument(
        "--watch_dir",
        default=str(DEFAULT_TEST_INPUT_DIR),
        help=f"Input folder watched when --watch is enabled. Default: {DEFAULT_TEST_INPUT_DIR}",
    )
    parser.add_argument(
        "--processed_dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help=(
            "Folder where checked images are moved when --watch is enabled. "
            f"Default: {DEFAULT_PROCESSED_DIR}"
        ),
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=1.0,
        help="Seconds between watch polls. Default: 1.0",
    )
    parser.add_argument(
        "--min_file_age",
        type=float,
        default=1.0,
        help="Minimum file age (seconds) before processing in watch mode. Default: 1.0",
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=0,
        help="Watch mode only: stop after N poll cycles. 0 means run forever.",
    )
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Default: WARNING",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON result instead of simple text output.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if args.watch:
        return _watch_loop(args)

    if not args.input:
        output = {
            "status": "ERROR",
            "error": "--input is required unless --watch is enabled.",
        }
        if args.json:
            print(json.dumps(output, ensure_ascii=True))
        else:
            print("ERROR")
        return 1

    output: Dict[str, Any] = {
        "match_label": NO_MATCH_LABEL,
        "match_distance": None,
        "threshold": float(args.threshold),
        "top_k": [],
        "status": NO_MATCH_LABEL,
    }

    try:
        db_encodings, db_labels, _meta = load_or_build_db(
            args.known_dir,
            args.cache_path,
            encode_mode=args.encoding_mode,
        )
        output = _run_match_for_input(
            args.input,
            db_encodings,
            db_labels,
            args.threshold,
            args.top_k,
            args.encoding_mode,
            args.multi_face,
        )
    except Exception as exc:  # pylint: disable=broad-except
        output = {
            "match_label": NO_MATCH_LABEL,
            "match_distance": None,
            "threshold": float(args.threshold),
            "top_k": [],
            "status": "ERROR",
            "error": str(exc),
        }
        if args.json:
            print(json.dumps(output, ensure_ascii=True))
        else:
            print("ERROR")
        return 1

    if args.json:
        print(json.dumps(output, ensure_ascii=True))
    else:
        print(_format_simple_output(output, args.multi_face))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
