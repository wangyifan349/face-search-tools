# SPDX-License-Identifier: AGPL-3.0-only
"""
Dlib + FAISS 1:N face search, packaged as one Flask file.

Overview
--------
This program is intended for a simple open-source face-search demo/service. It
scans a local gallery folder, encodes every detected face with dlib through the
face_recognition package, builds an exact FAISS inner-product index, and then
searches an uploaded query image against all indexed faces.

Folders
-------
- Gallery images: ./face_database
- Temporary query uploads: ./runtime_uploads

Search order
------------
All gallery and query embeddings are L2-normalized before FAISS insertion/search.
FAISS IndexFlatIP therefore ranks by cosine similarity. Returned results are
sorted from highest cosine similarity to lowest cosine similarity.

Returned metrics
----------------
The API/UI deliberately returns several metrics with different meanings:

    cosine_similarity       = dot(unit_query, unit_gallery)
    cosine_distance         = 1 - cosine_similarity
    normalized_l2_distance  = ||unit_query - unit_gallery||_2
                            = sqrt(2 - 2 * cosine_similarity)
    euclidean_distance      = ||raw_query - raw_gallery||_2

The "euclidean_distance" field is the raw 128D dlib/face_recognition Euclidean
distance. The "normalized_l2_distance" field is a different distance computed
after L2-normalization. They should not be mixed.

Threshold
---------
DEFAULT_MATCH_THRESHOLD is a cosine-similarity threshold, not the usual
face_recognition Euclidean tolerance. The default value 0.82 is close to the
normalized-vector equivalent of Euclidean distance 0.60:

    cosine = 1 - 0.60^2 / 2 = 0.82

Accuracy policy
---------------
Gallery encoding uses a higher jitter count for stable database vectors. Query
uploads use QUERY_FACE_ENCODING_NUM_JITTERS = 5, which is slower than the dlib
default but keeps upload-side search practical.

Run
---
    python dlib_faiss_flask_1n_single_file.py

Open
----
    https://localhost:5000

API example
-----------
    curl -k -X POST https://localhost:5000/api/search \
      -F "image=@/path/to/query.jpg" \
      -F "top_k=0" \
      -F "threshold=0.82"

License
-------
GNU Affero General Public License v3.0 only (AGPL-3.0-only).
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dlib
import faiss
import face_recognition
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_FOLDER = Path(__file__).resolve().parent
DATABASE_FOLDER = APP_FOLDER / "face_database"
UPLOAD_FOLDER = APP_FOLDER / "runtime_uploads"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EMBEDDING_DIMENSION = 128

# top_k=0 means return all indexed faces.
DEFAULT_TOP_K = 0

# This is a cosine-similarity threshold, not a face_recognition Euclidean
# distance threshold. 0.82 is roughly equivalent to Euclidean distance <= 0.60
# after L2 normalization: cosine = 1 - distance^2 / 2.
DEFAULT_MATCH_THRESHOLD = 0.82

# Upload/query limits.
MAX_UPLOAD_SIZE_MB = 64

# dlib / face_recognition parameters.
FACE_ENCODING_MODEL = "large"
FACE_DETECTION_UPSAMPLE_TIMES = 1
# Higher jitter count makes dlib compute several slightly perturbed encodings and average them.
# Gallery indexing can be slower because it is usually done once at startup.
FACE_ENCODING_NUM_JITTERS_CPU = 10
FACE_ENCODING_NUM_JITTERS_CUDA = 16

# Uploaded query images are encoded with a separate jitter count. This keeps query
# results more stable than the dlib default while avoiding very slow uploads.
QUERY_FACE_ENCODING_NUM_JITTERS = 5

# In high-accuracy mode, FAISS is still used as the exact vector index, but final
# returned scores and ranking are recomputed from stored float64 vectors. This
# avoids UI/API values being limited by FAISS float32 output precision.
HIGH_ACCURACY_EXACT_RERANK_ALL = True
# -----------------------------------------------------------------------------
# Runtime capability detection
# -----------------------------------------------------------------------------
def get_dlib_cuda_device_count() -> int:
    """Return visible CUDA device count without crashing on CPU-only dlib builds."""
    try:
        if not bool(getattr(dlib, "DLIB_USE_CUDA", False)):
            return 0
        return int(dlib.cuda.get_num_devices())
    except Exception as error:  # pragma: no cover - hardware/environment dependent
        print(f"[face-search] CUDA device check failed: {error}", flush=True)
        return 0

DLIB_VERSION = getattr(dlib, "__version__", "unknown")
IS_DLIB_COMPILED_WITH_CUDA = bool(getattr(dlib, "DLIB_USE_CUDA", False))
DLIB_CUDA_DEVICE_COUNT = get_dlib_cuda_device_count()
IS_CUDA_RUNTIME_AVAILABLE = IS_DLIB_COMPILED_WITH_CUDA and DLIB_CUDA_DEVICE_COUNT > 0

# Use CNN only when CUDA is actually usable. HOG is much safer for CPU deployment.
FACE_DETECTION_MODEL = "cnn" if IS_CUDA_RUNTIME_AVAILABLE else "hog"
FACE_ENCODING_NUM_JITTERS = (
    FACE_ENCODING_NUM_JITTERS_CUDA if IS_CUDA_RUNTIME_AVAILABLE else FACE_ENCODING_NUM_JITTERS_CPU
)

FAISS_CPU_THREAD_COUNT = max(1, os.cpu_count() or 1)
faiss.omp_set_num_threads(FAISS_CPU_THREAD_COUNT)

DATABASE_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def print_runtime_configuration() -> None:
    """Print deployment diagnostics once at startup."""
    print(f"[face-search] dlib version: {DLIB_VERSION}", flush=True)
    print(f"[face-search] dlib compiled with CUDA: {IS_DLIB_COMPILED_WITH_CUDA}", flush=True)
    print(f"[face-search] CUDA device count: {DLIB_CUDA_DEVICE_COUNT}", flush=True)
    print(f"[face-search] CUDA runtime available: {IS_CUDA_RUNTIME_AVAILABLE}", flush=True)
    print(f"[face-search] selected detection model: {FACE_DETECTION_MODEL}", flush=True)
    print(f"[face-search] selected encoding model: {FACE_ENCODING_MODEL}", flush=True)
    print(f"[face-search] upsample times: {FACE_DETECTION_UPSAMPLE_TIMES}", flush=True)
    print(f"[face-search] gallery num jitters: {FACE_ENCODING_NUM_JITTERS}", flush=True)
    print(f"[face-search] query num jitters: {QUERY_FACE_ENCODING_NUM_JITTERS}", flush=True)
    print(f"[face-search] FAISS CPU threads: {FAISS_CPU_THREAD_COUNT}", flush=True)
    print(f"[face-search] database folder: {DATABASE_FOLDER}", flush=True)

def clipped_cosine(value: float) -> float:
    """Clip small floating-point overshoot from FAISS inner-product results."""
    return max(-1.0, min(1.0, float(value)))

def cosine_to_normalized_l2(cosine_similarity: float) -> float:
    """Convert cosine similarity to L2 distance for normalized vectors."""
    cosine_similarity = clipped_cosine(cosine_similarity)
    return math.sqrt(max(0.0, 2.0 - 2.0 * cosine_similarity))

def cosine_to_display_percent(cosine_similarity: float) -> float:
    """
    Convert cosine similarity into a simple UI score.

    The search and threshold logic always use raw cosine similarity. This display
    value is only for humans reading the table. Negative cosine values are shown
    as 0 rather than a negative percent.
    """
    return max(0.0, min(100.0, clipped_cosine(cosine_similarity) * 100.0))

@dataclass(frozen=True)
class IndexBuildReport:
    """Summary of one startup database scan and FAISS index build."""

    database_folder: str
    scanned_image_count: int
    indexed_face_count: int
    failed_image_count: int
    skipped_image_count: int
    index_type: str
    metric: str
    faiss_thread_count: int
    num_jitters: int
    high_accuracy_exact_rerank_all: bool

class FaceSearchEngine:
    """Exact cosine-similarity face search engine using FAISS IndexFlatIP."""

    def __init__(self) -> None:
        self._state_lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._raw_embeddings_f64: Optional[np.ndarray] = None
        self._normalized_embeddings_f64: Optional[np.ndarray] = None
        self._face_metadata: List[Dict[str, Any]] = []
        self._latest_report = IndexBuildReport(
            database_folder=str(DATABASE_FOLDER),
            scanned_image_count=0,
            indexed_face_count=0,
            failed_image_count=0,
            skipped_image_count=0,
            index_type="faiss.IndexFlatIP exact exhaustive search",
            metric="cosine_similarity over L2-normalized dlib embeddings",
            faiss_thread_count=FAISS_CPU_THREAD_COUNT,
            num_jitters=FACE_ENCODING_NUM_JITTERS,
            high_accuracy_exact_rerank_all=HIGH_ACCURACY_EXACT_RERANK_ALL,
        )

    @staticmethod
    def _as_raw_embedding(face_embedding: np.ndarray) -> np.ndarray:
        """Return one raw 128D dlib embedding as float64 for precise distances."""
        embedding = np.asarray(face_embedding, dtype=np.float64).reshape(1, EMBEDDING_DIMENSION)
        if not np.all(np.isfinite(embedding)):
            raise ValueError("Invalid face embedding values")
        return np.ascontiguousarray(embedding)

    @staticmethod
    def _normalize_embedding(face_embedding: np.ndarray) -> np.ndarray:
        """L2-normalize one 128D embedding as float64 before cosine scoring."""
        embedding = np.asarray(face_embedding, dtype=np.float64).reshape(1, EMBEDDING_DIMENSION)
        embedding_norm = float(np.linalg.norm(embedding, ord=2))
        if not np.isfinite(embedding_norm) or embedding_norm <= 0.0:
            raise ValueError("Invalid face embedding norm")
        normalized_embedding = embedding / embedding_norm
        return np.ascontiguousarray(normalized_embedding)

    @staticmethod
    def _face_location_to_box(face_location: Tuple[int, int, int, int]) -> Dict[str, int]:
        """Convert face_recognition's (top, right, bottom, left) tuple to JSON."""
        top, right, bottom, left = face_location
        return {
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "left": int(left),
            "width": int(max(0, right - left)),
            "height": int(max(0, bottom - top)),
        }

    @staticmethod
    def _face_area(face_location: Tuple[int, int, int, int]) -> int:
        """Return detected face area so the largest query face can be selected."""
        top, right, bottom, left = face_location
        return max(0, right - left) * max(0, bottom - top)

    @staticmethod
    def _list_database_images(database_folder: Path) -> List[Path]:
        """Find supported image files under the database folder."""
        image_paths: List[Path] = []
        if not database_folder.is_dir():
            return image_paths

        for current_folder, _, file_names in os.walk(database_folder):
            for file_name in sorted(file_names):
                image_path = Path(current_folder) / file_name
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(image_path)

        return sorted(image_paths, key=lambda path: path.as_posix())

    def _extract_face_records_from_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """Detect and encode every face in one gallery image."""
        image_array = face_recognition.load_image_file(str(image_path))

        detect_started_at = time.perf_counter()
        face_locations = face_recognition.face_locations(
            image_array,
            number_of_times_to_upsample=FACE_DETECTION_UPSAMPLE_TIMES,
            model=FACE_DETECTION_MODEL,
        )
        detect_seconds = time.perf_counter() - detect_started_at
        print(
            f"[face-search] detected {len(face_locations)} face(s) in {detect_seconds:.2f}s: {image_path.name}",
            flush=True,
        )

        if not face_locations:
            return []

        encode_started_at = time.perf_counter()
        face_encodings = face_recognition.face_encodings(
            image_array,
            known_face_locations=face_locations,
            num_jitters=FACE_ENCODING_NUM_JITTERS,
            model=FACE_ENCODING_MODEL,
        )
        encode_seconds = time.perf_counter() - encode_started_at
        print(
            f"[face-search] encoded {len(face_encodings)} face(s) in {encode_seconds:.2f}s: {image_path.name}",
            flush=True,
        )

        face_records: List[Dict[str, Any]] = []
        for face_index, face_embedding in enumerate(face_encodings):
            raw_embedding = self._as_raw_embedding(face_embedding)
            normalized_embedding = self._normalize_embedding(raw_embedding)
            face_records.append(
                {
                    "face_index": face_index,
                    "face_box": self._face_location_to_box(face_locations[face_index]),
                    "raw_embedding": raw_embedding,
                    "normalized_embedding": normalized_embedding,
                }
            )
        return face_records

    def rebuild_index(self, database_folder: Path = DATABASE_FOLDER) -> IndexBuildReport:
        """Re-scan gallery images, encode faces, and atomically replace the FAISS index."""
        with self._build_lock:
            database_folder = Path(database_folder).resolve()
            database_folder.mkdir(parents=True, exist_ok=True)

            scanned_image_count = 0
            failed_image_count = 0
            skipped_image_count = 0
            raw_embeddings: List[np.ndarray] = []
            normalized_embeddings: List[np.ndarray] = []
            face_metadata: List[Dict[str, Any]] = []
            exact_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

            image_paths = self._list_database_images(database_folder)
            print(f"[face-search] database folder: {database_folder}", flush=True)
            print(f"[face-search] found {len(image_paths)} gallery image(s).", flush=True)

            if not image_paths:
                print(
                    "[face-search] no gallery images found. Put images into face_database and restart.",
                    flush=True,
                )

            for image_number, image_path in enumerate(image_paths, start=1):
                scanned_image_count += 1
                relative_path = image_path.relative_to(database_folder).as_posix()
                print(
                    f"[face-search] [{image_number}/{len(image_paths)}] encoding: {relative_path}",
                    flush=True,
                )

                try:
                    face_records = self._extract_face_records_from_image(image_path)
                    if not face_records:
                        skipped_image_count += 1
                        print(
                            f"[face-search] [{image_number}/{len(image_paths)}] no face detected: {relative_path}",
                            flush=True,
                        )
                        continue

                    for face_record in face_records:
                        vector_id = len(face_metadata)
                        raw_embeddings.append(face_record["raw_embedding"])
                        normalized_embeddings.append(face_record["normalized_embedding"])
                        face_metadata.append(
                            {
                                "vector_id": vector_id,
                                "file_name": image_path.name,
                                "relative_path": relative_path,
                                "face_index": face_record["face_index"],
                                "face_box": face_record["face_box"],
                            }
                        )
                except Exception as error:
                    failed_image_count += 1
                    print(f"[face-search] failed to process {relative_path}: {error}", flush=True)

            if normalized_embeddings:
                raw_embedding_matrix_f64 = np.ascontiguousarray(np.vstack(raw_embeddings), dtype=np.float64)
                normalized_embedding_matrix_f64 = np.ascontiguousarray(
                    np.vstack(normalized_embeddings),
                    dtype=np.float64,
                )
                normalized_embedding_matrix_f32 = np.ascontiguousarray(
                    normalized_embedding_matrix_f64.astype(np.float32),
                    dtype=np.float32,
                )
                exact_index.add(normalized_embedding_matrix_f32)
                print(f"[face-search] FAISS insertion finished. ntotal={exact_index.ntotal}", flush=True)
            else:
                raw_embedding_matrix_f64 = None
                normalized_embedding_matrix_f64 = None
                print("[face-search] FAISS index is empty; no valid face vector was produced.", flush=True)

            report = IndexBuildReport(
                database_folder=str(database_folder),
                scanned_image_count=scanned_image_count,
                indexed_face_count=len(face_metadata),
                failed_image_count=failed_image_count,
                skipped_image_count=skipped_image_count,
                index_type="faiss.IndexFlatIP exact exhaustive search",
                metric="cosine_similarity over L2-normalized dlib embeddings",
                faiss_thread_count=FAISS_CPU_THREAD_COUNT,
                num_jitters=FACE_ENCODING_NUM_JITTERS,
                high_accuracy_exact_rerank_all=HIGH_ACCURACY_EXACT_RERANK_ALL,
            )

            with self._state_lock:
                self._faiss_index = exact_index
                self._raw_embeddings_f64 = raw_embedding_matrix_f64
                self._normalized_embeddings_f64 = normalized_embedding_matrix_f64
                self._face_metadata = face_metadata
                self._latest_report = report

            print(
                "[face-search] build report: "
                f"scanned={report.scanned_image_count}, "
                f"indexed_faces={report.indexed_face_count}, "
                f"skipped_no_face={report.skipped_image_count}, "
                f"failed={report.failed_image_count}",
                flush=True,
            )
            return report

    def get_snapshot(
        self,
    ) -> Tuple[
        Optional[faiss.IndexFlatIP],
        Optional[np.ndarray],
        Optional[np.ndarray],
        List[Dict[str, Any]],
        IndexBuildReport,
    ]:
        """Return a short-lived read snapshot for concurrent search requests."""
        with self._state_lock:
            return (
                self._faiss_index,
                self._raw_embeddings_f64,
                self._normalized_embeddings_f64,
                list(self._face_metadata),
                self._latest_report,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Return current database, model, and FAISS status for UI/API display."""
        faiss_index, raw_embeddings_f64, normalized_embeddings_f64, face_metadata, report = self.get_snapshot()
        database_folder = Path(report.database_folder)
        gallery_image_count = len(self._list_database_images(database_folder))
        return {
            **asdict(report),
            "database_folder_exists": database_folder.is_dir(),
            "gallery_image_count": gallery_image_count,
            "faiss_ntotal": int(faiss_index.ntotal) if faiss_index is not None else 0,
            "metadata_count": len(face_metadata),
            "detection_model": FACE_DETECTION_MODEL,
            "encoding_model": FACE_ENCODING_MODEL,
            "upsample_times": FACE_DETECTION_UPSAMPLE_TIMES,
            "gallery_num_jitters": FACE_ENCODING_NUM_JITTERS,
            "query_num_jitters": QUERY_FACE_ENCODING_NUM_JITTERS,
            "default_top_k": DEFAULT_TOP_K,
            "default_threshold": DEFAULT_MATCH_THRESHOLD,
            "threshold_metric": "cosine_similarity",
            "sort_order": "cosine_similarity_descending",
            "score_precision": "float64 final scoring",
            "returns_raw_euclidean_distance": True,
            "max_upload_mb": MAX_UPLOAD_SIZE_MB,
            "dlib_version": DLIB_VERSION,
            "is_dlib_compiled_with_cuda": IS_DLIB_COMPILED_WITH_CUDA,
            "cuda_device_count": DLIB_CUDA_DEVICE_COUNT,
            "is_cuda_runtime_available": IS_CUDA_RUNTIME_AVAILABLE,
        }

    def search_uploaded_image(self, image_path: Path, top_k: int, threshold: float) -> Dict[str, Any]:
        """Encode one query image and return exact cosine matches sorted high to low."""
        faiss_index, raw_embeddings_f64, normalized_embeddings_f64, face_metadata, report = self.get_snapshot()
        if (
            faiss_index is None
            or faiss_index.ntotal <= 0
            or raw_embeddings_f64 is None
            or normalized_embeddings_f64 is None
        ):
            raise RuntimeError(
                "FAISS index is empty. Put gallery images into "
                f"{DATABASE_FOLDER} and restart the service."
            )

        query_image = face_recognition.load_image_file(str(image_path))

        query_detect_started_at = time.perf_counter()
        query_face_locations = face_recognition.face_locations(
            query_image,
            number_of_times_to_upsample=FACE_DETECTION_UPSAMPLE_TIMES,
            model=FACE_DETECTION_MODEL,
        )
        query_detect_seconds = time.perf_counter() - query_detect_started_at
        print(
            f"[face-search] query detection finished: "
            f"faces={len(query_face_locations)}, seconds={query_detect_seconds:.2f}",
            flush=True,
        )

        if not query_face_locations:
            raise ValueError("No face detected in query image")

        selected_face_location = max(query_face_locations, key=self._face_area)

        query_encode_started_at = time.perf_counter()
        query_face_encodings = face_recognition.face_encodings(
            query_image,
            known_face_locations=[selected_face_location],
            num_jitters=QUERY_FACE_ENCODING_NUM_JITTERS,
            model=FACE_ENCODING_MODEL,
        )
        query_encode_seconds = time.perf_counter() - query_encode_started_at
        print(
            f"[face-search] query encoding finished: "
            f"encodings={len(query_face_encodings)}, seconds={query_encode_seconds:.2f}",
            flush=True,
        )

        if not query_face_encodings:
            raise ValueError("Failed to extract query face embedding")

        query_raw_embedding = self._as_raw_embedding(query_face_encodings[0])
        query_normalized_embedding = self._normalize_embedding(query_raw_embedding)
        query_normalized_embedding_f32 = np.ascontiguousarray(
            query_normalized_embedding.astype(np.float32),
            dtype=np.float32,
        )

        indexed_face_count = int(faiss_index.ntotal)
        requested_top_k = int(top_k)
        returned_count = indexed_face_count if requested_top_k <= 0 else min(requested_top_k, indexed_face_count)

        # FAISS is still queried so the service keeps using the exact FAISS index.
        # Final metrics below are recomputed in float64 for display/API accuracy.
        faiss_search_started_at = time.perf_counter()
        faiss_similarities, faiss_indices = faiss_index.search(query_normalized_embedding_f32, indexed_face_count)
        faiss_search_seconds = time.perf_counter() - faiss_search_started_at
        print(
            f"[face-search] FAISS exhaustive search finished: candidates={indexed_face_count}, "
            f"seconds={faiss_search_seconds:.4f}",
            flush=True,
        )

        precise_score_started_at = time.perf_counter()
        cosine_scores = normalized_embeddings_f64 @ query_normalized_embedding.reshape(EMBEDDING_DIMENSION)
        cosine_scores = np.clip(cosine_scores.astype(np.float64, copy=False), -1.0, 1.0)
        raw_deltas = raw_embeddings_f64 - query_raw_embedding.reshape(EMBEDDING_DIMENSION)
        raw_euclidean_distances = np.linalg.norm(raw_deltas, axis=1)
        normalized_l2_distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine_scores))

        # Primary order: cosine similarity descending. Tie-breaker: lower raw Euclidean
        # distance. Stable sorting keeps output deterministic for nearly identical scores.
        sorted_indices = sorted(
            range(indexed_face_count),
            key=lambda idx: (
                -float(cosine_scores[idx]),
                float(raw_euclidean_distances[idx]),
                str(face_metadata[idx].get("relative_path", "")),
                int(face_metadata[idx].get("face_index", 0)),
            ),
        )
        final_indices = sorted_indices[:returned_count]
        precise_score_seconds = time.perf_counter() - precise_score_started_at
        print(
            f"[face-search] float64 scoring/rerank finished: returned={returned_count}, "
            f"seconds={precise_score_seconds:.4f}",
            flush=True,
        )

        results: List[Dict[str, Any]] = []
        for rank_index, vector_index in enumerate(final_indices, start=1):
            metadata = dict(face_metadata[int(vector_index)])
            cosine_similarity = clipped_cosine(float(cosine_scores[vector_index]))
            cosine_distance = 1.0 - cosine_similarity
            normalized_l2_distance = float(normalized_l2_distances[vector_index])
            euclidean_distance = float(raw_euclidean_distances[vector_index])
            similarity_percent = cosine_to_display_percent(cosine_similarity)

            relative_path = str(metadata.get("relative_path", ""))
            image_url = url_for("serve_database_image", relative_path=relative_path) if relative_path else None
            download_url = url_for("download_database_image", relative_path=relative_path) if relative_path else None

            results.append(
                {
                    "rank": rank_index,
                    "vector_id": metadata.get("vector_id"),
                    "file_name": metadata.get("file_name"),
                    "relative_path": relative_path,
                    "image_url": image_url,
                    "download_url": download_url,
                    "face_index": metadata.get("face_index"),
                    "face_box": metadata.get("face_box"),
                    "cosine_similarity": round(cosine_similarity, 10),
                    "similarity_percent": round(similarity_percent, 4),
                    "cosine_distance": round(cosine_distance, 10),
                    "euclidean_distance": round(euclidean_distance, 10),
                    "normalized_l2_distance": round(normalized_l2_distance, 10),
                    "threshold": round(float(threshold), 10),
                    "threshold_metric": "cosine_similarity",
                    "is_match": bool(cosine_similarity >= threshold),
                    "score_precision": "float64",
                }
            )

        for left, right in zip(results, results[1:]):
            if left["cosine_similarity"] < right["cosine_similarity"]:
                raise RuntimeError("Internal error: search results are not sorted by cosine similarity descending")

        if results and int(faiss_indices[0][0]) != int(final_indices[0]):
            print(
                "[face-search] note: float64 rerank changed the top result after FAISS float32 scoring.",
                flush=True,
            )

        return {
            "success": True,
            "query": {
                "file_name": image_path.name,
                "detected_face_count": len(query_face_locations),
                "selected_face_box": self._face_location_to_box(selected_face_location),
            },
            "search": {
                "metric": "cosine_similarity",
                "metric_range": "[-1, 1]",
                "sort_order": "descending",
                "requested_top_k": requested_top_k,
                "actual_top_k": returned_count,
                "returned_count": len(results),
                "match_count": sum(1 for result in results if result["is_match"]),
                "threshold": round(float(threshold), 8),
                "threshold_metric": "cosine_similarity",
                "score_precision": "float64",
                "distance_fields": {
                    "euclidean_distance": "raw dlib/face_recognition 128D Euclidean distance",
                    "normalized_l2_distance": "Euclidean distance after L2-normalization",
                    "cosine_distance": "1 - cosine_similarity",
                },
                "index_type": report.index_type,
                "indexed_face_count": report.indexed_face_count,
                "scanned_image_count": report.scanned_image_count,
                "faiss_search_seconds": round(faiss_search_seconds, 6),
                "precise_score_seconds": round(precise_score_seconds, 6),
                "query_detection_seconds": round(query_detect_seconds, 6),
                "query_encoding_seconds": round(query_encode_seconds, 6),
                "gallery_num_jitters": FACE_ENCODING_NUM_JITTERS,
                "query_num_jitters": QUERY_FACE_ENCODING_NUM_JITTERS,
            },
            "results": results,
        }

face_search_engine = FaceSearchEngine()
# -----------------------------------------------------------------------------
# Minimal web UI
# -----------------------------------------------------------------------------
HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Search</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --bs-body-font-family: inherit;
      --bs-body-bg: #f3efdf;
      --bs-body-color: #1f2a20;
      --bs-border-color: #bdb49d;
      --bs-link-color: #2e5b46;
      --bs-link-hover-color: #203f31;
      --bs-primary: #2e5b46;
    }
    html, body {
      min-height: 100%;
      background: #f3efdf;
      color: #1f2a20;
      font-size: 19px;
    }
    body, input, button, select, textarea {
      font-family: inherit !important;
    }
    a { color: #2e5b46; }
    a:hover { color: #203f31; }
    .page-shell {
      width: 100%;
      min-height: 100vh;
      padding: 22px;
    }
    h1 {
      font-size: 1.8rem;
      margin: 0 0 16px;
      font-weight: 650;
    }
    .search-bar {
      width: 100%;
      background: #e7e0ce;
      border: 1px solid #bdb49d;
      border-radius: 10px;
      padding: 16px;
    }
    .form-label {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 6px;
    }
    .form-control, .btn {
      font-size: 1rem;
      min-height: 48px;
    }
    .form-control {
      background-color: #fffdf4;
      color: #1f2a20;
      border-color: #bdb49d;
    }
    .form-control:focus {
      border-color: #55775a;
      box-shadow: 0 0 0 0.2rem rgba(85, 119, 90, 0.22);
    }
    .btn-search {
      --bs-btn-color: #fffdf4;
      --bs-btn-bg: #3d654d;
      --bs-btn-border-color: #3d654d;
      --bs-btn-hover-color: #fffdf4;
      --bs-btn-hover-bg: #2f523d;
      --bs-btn-hover-border-color: #2f523d;
      --bs-btn-focus-shadow-rgb: 85, 119, 90;
      --bs-btn-active-color: #fffdf4;
      --bs-btn-active-bg: #274634;
      --bs-btn-active-border-color: #274634;
      font-weight: 650;
    }
    .hint, .status { color: #566050; }
    .hint {
      margin: 14px 0 10px;
      font-size: 0.98rem;
    }
    .status {
      min-height: 28px;
      margin: 10px 0;
      font-size: 1rem;
    }
    .error {
      color: #8a2d20;
      font-weight: 700;
    }
    .summary {
      width: 100%;
      margin: 12px 0;
      padding: 12px 14px;
      background: #e7e0ce;
      border: 1px solid #bdb49d;
      border-radius: 10px;
    }
    .table-wrap {
      width: 100%;
      overflow-x: auto;
      margin-top: 12px;
      border: 1px solid #bdb49d;
      border-radius: 10px;
      background: #f8f4e8;
    }
    table {
      width: 100%;
      margin: 0;
      font-size: 1rem;
      vertical-align: top;
    }
    thead th {
      background: #ddd4bd;
      color: #1f2a20;
      white-space: nowrap;
      border-bottom: 1px solid #bdb49d !important;
    }
    td, th {
      padding: 12px !important;
      vertical-align: top;
      border-color: #d1c6ad !important;
    }
    .table-hover > tbody > tr:hover > * {
      background-color: #eee8d8;
      color: #1f2a20;
    }
    .preview {
      width: 380px;
      max-width: 38vw;
      max-height: 380px;
      object-fit: contain;
      display: block;
      background: #ddd4bd;
      border: 1px solid #bdb49d;
      border-radius: 8px;
    }
    .path {
      max-width: 440px;
      word-break: break-all;
      font-size: 0.92rem;
    }
    .links {
      margin-top: 8px;
      font-size: 0.95rem;
    }
    .links a { margin-right: 14px; }
    .match-yes {
      color: #23543e;
      font-weight: 700;
    }
    .match-no {
      color: #7a3b20;
      font-weight: 700;
    }
    @media (max-width: 900px) {
      .page-shell { padding: 14px; }
      html, body { font-size: 18px; }
      .preview { width: 100%; max-width: 100%; max-height: 420px; }
      table, thead, tbody, tr, th, td { display: block; }
      thead { display: none; }
      tr { border-bottom: 1px solid #bdb49d; padding: 10px 0; }
      td { border: 0 !important; padding: 7px 12px !important; }
      td::before {
        content: attr(data-label);
        display: block;
        color: #566050;
        font-size: 0.85rem;
        margin-bottom: 3px;
      }
    }
  </style>
</head>
<body>
  <main class="page-shell">
    <h1>Face Search</h1>
    <form id="searchForm" class="search-bar row g-3 align-items-end">
      <div class="col-12 col-lg-5">
        <label for="imageInput" class="form-label">Query image</label>
        <input id="imageInput" name="image" class="form-control" type="file" accept="image/*" required>
      </div>
      <div class="col-6 col-lg-2">
        <label for="topKInput" class="form-label">Result count</label>
        <input id="topKInput" name="top_k" class="form-control" type="number" min="0" step="1" value="0">
      </div>
      <div class="col-6 col-lg-2">
        <label for="thresholdInput" class="form-label">Cosine threshold</label>
        <input id="thresholdInput" name="threshold" class="form-control" type="number" min="-1" max="1" step="0.01" value="0.82">
      </div>
      <div class="col-12 col-lg-3 d-grid">
        <button class="btn btn-search" type="submit">Search</button>
      </div>
    </form>
    <p class="hint">Set result count to 0 to return all indexed faces. Results are sorted by cosine similarity descending. Euclidean distance is the raw 128D dlib embedding distance.</p>
    <div id="status" class="status"></div>
    <div id="summary"></div>
    <div id="results"></div>
  </main>
  <script>
    const form = document.getElementById('searchForm');
    const statusEl = document.getElementById('status');
    const summaryEl = document.getElementById('summary');
    const resultsEl = document.getElementById('results');
    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? 'status error' : 'status';
    }
    function addTextCell(row, label, text, className = '') {
      const cell = document.createElement('td');
      cell.dataset.label = label;
      cell.textContent = text;
      if (className) cell.className = className;
      row.appendChild(cell);
      return cell;
    }
    function renderSummary(data) {
      summaryEl.innerHTML = '';
      const box = document.createElement('div');
      box.className = 'summary';
      box.textContent =
        `query faces: ${data.query.detected_face_count}; ` +
        `indexed faces: ${data.search.indexed_face_count}; ` +
        `returned: ${data.search.returned_count}; ` +
        `matches: ${data.search.match_count}; ` +
        `cosine threshold: ${data.search.threshold}; ` +
        `query jitters: ${data.search.query_num_jitters}; ` +
        `query encoding: ${data.search.query_encoding_seconds}s; ` +
        `FAISS: ${data.search.faiss_search_seconds}s; ` +
        `rerank: ${data.search.precise_score_seconds}s.`;
      summaryEl.appendChild(box);
    }
    function renderResults(results) {
      resultsEl.innerHTML = '';
      if (!results.length) {
        resultsEl.textContent = 'No results returned.';
        return;
      }
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      const table = document.createElement('table');
      table.className = 'table table-hover align-middle';
      const thead = document.createElement('thead');
      const headRow = document.createElement('tr');
      ['Rank', 'Image', 'Similarity', 'Cosine', 'Euclidean distance', 'Normalized L2', 'Cosine distance', 'Match', 'File'].forEach(title => {
        const th = document.createElement('th');
        th.textContent = title;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);
      const tbody = document.createElement('tbody');
      for (const item of results) {
        const row = document.createElement('tr');
        addTextCell(row, 'Rank', String(item.rank));
        const imageCell = document.createElement('td');
        imageCell.dataset.label = 'Image';
        if (item.image_url) {
          const link = document.createElement('a');
          link.href = item.image_url;
          link.target = '_blank';
          link.rel = 'noopener';
          const img = document.createElement('img');
          img.className = 'preview';
          img.src = item.image_url;
          img.alt = item.relative_path || item.file_name || 'matched image';
          link.appendChild(img);
          imageCell.appendChild(link);
        }
        row.appendChild(imageCell);
        addTextCell(row, 'Similarity', `${Number(item.similarity_percent).toFixed(2)}%`);
        addTextCell(row, 'Cosine', Number(item.cosine_similarity).toFixed(10));
        addTextCell(row, 'Euclidean distance', Number(item.euclidean_distance).toFixed(10));
        addTextCell(row, 'Normalized L2', Number(item.normalized_l2_distance).toFixed(10));
        addTextCell(row, 'Cosine distance', Number(item.cosine_distance).toFixed(10));
        addTextCell(row, 'Match', item.is_match ? 'Yes' : 'No', item.is_match ? 'match-yes' : 'match-no');
        const fileCell = document.createElement('td');
        fileCell.dataset.label = 'File';
        const pathDiv = document.createElement('div');
        pathDiv.className = 'path';
        pathDiv.textContent = item.relative_path || item.file_name || '';
        fileCell.appendChild(pathDiv);
        const linksDiv = document.createElement('div');
        linksDiv.className = 'links';
        if (item.image_url) {
          const openLink = document.createElement('a');
          openLink.href = item.image_url;
          openLink.target = '_blank';
          openLink.rel = 'noopener';
          openLink.textContent = 'Open';
          linksDiv.appendChild(openLink);
        }
        if (item.download_url) {
          const downloadLink = document.createElement('a');
          downloadLink.href = item.download_url;
          downloadLink.textContent = 'Download';
          linksDiv.appendChild(downloadLink);
        }
        fileCell.appendChild(linksDiv);
        row.appendChild(fileCell);
        tbody.appendChild(row);
      }
      table.appendChild(tbody);
      wrap.appendChild(table);
      resultsEl.appendChild(wrap);
    }
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      summaryEl.innerHTML = '';
      resultsEl.innerHTML = '';
      setStatus('Searching...');
      const formData = new FormData(form);
      try {
        const response = await fetch('/api/search', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || `HTTP ${response.status}`);
        setStatus('Search completed.');
        renderSummary(data);
        renderResults(data.results || []);
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    });
  </script>
</body>
</html>"""
# -----------------------------------------------------------------------------
# Request parsing and validation
# -----------------------------------------------------------------------------
def parse_top_k(raw_value: Optional[str]) -> int:
    """Parse the top_k form/API value. 0 means return all indexed faces."""
    if raw_value is None or str(raw_value).strip() == "":
        return DEFAULT_TOP_K

    try:
        parsed_value = int(str(raw_value).strip())
    except ValueError as error:
        raise ValueError("top_k must be an integer. Use 0 to return all results.") from error

    if parsed_value < 0:
        raise ValueError("top_k must be greater than or equal to 0")
    return parsed_value

def parse_threshold(raw_value: Optional[str]) -> float:
    """Parse and validate the cosine-similarity match threshold."""
    if raw_value is None or str(raw_value).strip() == "":
        return DEFAULT_MATCH_THRESHOLD

    try:
        parsed_value = float(str(raw_value).strip())
    except ValueError as error:
        raise ValueError("threshold must be a number") from error

    if parsed_value < -1.0 or parsed_value > 1.0:
        raise ValueError("threshold must be between -1.0 and 1.0 for cosine similarity")
    return parsed_value

def validate_image_filename(file_name: str) -> str:
    """Return a safe query image filename or raise if the extension is unsupported."""
    safe_name = secure_filename(file_name) or "image.jpg"
    file_extension = Path(safe_name).suffix.lower()
    if file_extension not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {file_extension}")
    return safe_name

def save_query_image() -> Path:
    """Save the uploaded query image into the temporary upload folder."""
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        raise ValueError("Missing query image. Upload using form field name: image")

    safe_name = validate_image_filename(uploaded_file.filename)
    saved_path = UPLOAD_FOLDER / f"query_{uuid.uuid4().hex}{Path(safe_name).suffix.lower()}"
    uploaded_file.save(saved_path)
    return saved_path
# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index_page():
    """Serve the minimal search-only web interface."""
    return render_template_string(HTML_PAGE)

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return current database and FAISS status."""
    return jsonify(face_search_engine.get_stats())

@app.route("/api/health", methods=["GET"])
def api_health():
    """Return a lightweight readiness response."""
    stats = face_search_engine.get_stats()
    return jsonify(
        {
            "success": True,
            "ready": stats["indexed_face_count"] > 0,
            "indexed_face_count": stats["indexed_face_count"],
            "gallery_image_count": stats["gallery_image_count"],
            "metric": stats["metric"],
            "threshold_metric": stats["threshold_metric"],
            "sort_order": stats["sort_order"],
            "index_type": stats["index_type"],
        }
    )

@app.route("/api/search", methods=["POST"])
def api_search():
    """Search one uploaded query image against the startup-built FAISS database."""
    saved_path: Optional[Path] = None
    try:
        top_k = parse_top_k(request.form.get("top_k"))
        threshold = parse_threshold(request.form.get("threshold"))
        saved_path = save_query_image()
        return jsonify(face_search_engine.search_uploaded_image(saved_path, top_k, threshold))
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400
    finally:
        if saved_path is not None:
            try:
                saved_path.unlink(missing_ok=True)
            except Exception:
                pass

@app.route("/database-image/<path:relative_path>", methods=["GET"])
def serve_database_image(relative_path: str):
    """Serve matched gallery images from face_database for result display."""
    return send_from_directory(DATABASE_FOLDER, relative_path)

@app.route("/database-image-download/<path:relative_path>", methods=["GET"])
def download_database_image(relative_path: str):
    """Download a matched gallery image as an attachment."""
    return send_from_directory(DATABASE_FOLDER, relative_path, as_attachment=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error):  # noqa: ARG001 - Flask passes the exception object.
    """Return a clear JSON error when the query image exceeds the size limit."""
    return jsonify({"success": False, "error": f"Uploaded file is larger than {MAX_UPLOAD_SIZE_MB} MB"}), 413

if __name__ == "__main__":
    print_runtime_configuration()
    print("[face-search] starting service and building initial FAISS index...", flush=True)
    startup_report = face_search_engine.rebuild_index(DATABASE_FOLDER)
    print(
        "[face-search] ready: "
        f"{startup_report.indexed_face_count} face vector(s) from "
        f"{startup_report.scanned_image_count} image(s). "
        f"Skipped no-face: {startup_report.skipped_image_count}. "
        f"Failed: {startup_report.failed_image_count}. "
        "Open https://localhost:5000",
        flush=True,
    )
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context="adhoc")
