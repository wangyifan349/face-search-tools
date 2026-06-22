"""
Local 1:N face search with dlib, face_recognition, and FAISS.
The script asks for a face database folder, builds an in-memory FAISS index,
then repeatedly asks for query image paths and prints the top matching faces.
Install:
    pip install numpy dlib face_recognition faiss-cpu
"""

import math
import os
import sys
import time
from pathlib import Path
import dlib
import faiss
import face_recognition
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # Supported image formats
EMBEDDING_DIMENSION = 128  # face_recognition returns 128D face embeddings
TOP_K = 10  # Number of results to print
MATCH_THRESHOLD = 0.80  # Cosine threshold, larger is stricter
ENCODING_MODEL = "large"  # dlib large encoding model
UPSAMPLE_TIMES = 0  # No upsampling, not optimized for small faces
CPU_GALLERY_JITTERS = 8  # Gallery encoding jitter count on CPU
CUDA_GALLERY_JITTERS = 20  # Gallery encoding jitter count on CUDA
CPU_QUERY_JITTERS = 5  # Query encoding jitter count on CPU
CUDA_QUERY_JITTERS = 10  # Query encoding jitter count on CUDA

dlib_version = getattr(dlib, "__version__", "unknown")  # dlib version string
dlib_has_cuda = bool(getattr(dlib, "DLIB_USE_CUDA", False))  # Whether dlib was compiled with CUDA

cuda_device_count = 0  # Visible CUDA device count
try:
    if dlib_has_cuda:
        cuda_device_count = int(dlib.cuda.get_num_devices())
except Exception as error:
    print(f"[WARN] CUDA check failed: {error}")

cuda_is_available = dlib_has_cuda and cuda_device_count > 0  # True only when CUDA can actually run
detection_model = "cnn" if cuda_is_available else "hog"  # CNN with CUDA, HOG otherwise
gallery_jitters = CUDA_GALLERY_JITTERS if cuda_is_available else CPU_GALLERY_JITTERS  # Gallery precision setting
query_jitters = CUDA_QUERY_JITTERS if cuda_is_available else CPU_QUERY_JITTERS  # Query precision setting
faiss_thread_count = max(1, os.cpu_count() or 1)  # FAISS CPU thread count
faiss.omp_set_num_threads(faiss_thread_count)  # Apply FAISS thread setting

def clean_path(raw_path):
    return raw_path.strip().strip('"').strip("'")  # Clean common terminal quotes

def format_seconds(seconds):
    return f"{seconds:.4f}s"  # Keep timing output short

def get_face_area(face_location):
    top, right, bottom, left = face_location  # face_recognition box format
    width = max(0, right - left)  # Face box width
    height = max(0, bottom - top)  # Face box height
    return width * height  # Face area

def face_location_to_box(face_location):
    top, right, bottom, left = face_location  # face_recognition box format
    width = max(0, right - left)  # Face box width
    height = max(0, bottom - top)  # Face box height
    box = {}  # Plain dictionary for readable output
    box["top"] = int(top)
    box["right"] = int(right)
    box["bottom"] = int(bottom)
    box["left"] = int(left)
    box["width"] = int(width)
    box["height"] = int(height)
    return box

def to_raw_embedding(face_embedding):
    embedding = np.asarray(face_embedding, dtype=np.float64)  # Keep raw embedding in float64
    embedding = embedding.reshape(1, EMBEDDING_DIMENSION)  # Force shape to 1 x 128
    if not np.all(np.isfinite(embedding)):
        raise ValueError("Invalid face embedding values")
    return np.ascontiguousarray(embedding)  # FAISS and NumPy prefer contiguous arrays

def to_normalized_embedding(face_embedding):
    embedding = np.asarray(face_embedding, dtype=np.float64)  # Convert to float64
    embedding = embedding.reshape(1, EMBEDDING_DIMENSION)  # Force shape to 1 x 128
    embedding_norm = float(np.linalg.norm(embedding, ord=2))  # L2 norm
    if not np.isfinite(embedding_norm) or embedding_norm <= 0.0:
        raise ValueError("Invalid face embedding norm")
    normalized_embedding = embedding / embedding_norm  # Unit vector for cosine search
    return np.ascontiguousarray(normalized_embedding)  # Keep memory contiguous

def ask_database_folder():
    while True:
        raw_database_path = input("Enter face database folder path: ")
        database_path = clean_path(raw_database_path)
        if not database_path:
            print("Database folder path cannot be empty.")
            continue
        database_folder = Path(database_path).expanduser().resolve()
        if not database_folder.is_dir():
            print(f"Invalid folder path: {database_folder}")
            continue
        return database_folder

def list_image_paths(database_folder):
    image_paths = []  # Store all supported gallery image paths
    for current_folder, folder_names, file_names in os.walk(database_folder):
        folder_names.sort()  # Stable traversal order
        file_names.sort()  # Stable file order
        for file_name in file_names:
            image_path = Path(current_folder) / file_name
            file_extension = image_path.suffix.lower()
            if file_extension in IMAGE_EXTENSIONS:
                image_paths.append(image_path)
    image_paths.sort(key=lambda path: path.as_posix())  # Stable final order
    return image_paths

def print_runtime_info(database_folder):
    print()
    print("[face-search] runtime configuration")
    print(f"  dlib version: {dlib_version}")
    print(f"  dlib compiled with CUDA: {dlib_has_cuda}")
    print(f"  CUDA device count: {cuda_device_count}")
    print(f"  CUDA runtime available: {cuda_is_available}")
    print(f"  detection model: {detection_model}")
    print(f"  encoding model: {ENCODING_MODEL}")
    print(f"  upsample times: {UPSAMPLE_TIMES}")
    print(f"  gallery jitters: {gallery_jitters}")
    print(f"  query jitters: {query_jitters}")
    print(f"  FAISS CPU threads: {faiss_thread_count}")
    print(f"  database folder: {database_folder}")
    print(f"  top_k: {TOP_K}")
    print(f"  threshold: cosine_similarity >= {MATCH_THRESHOLD}")
    print()

def extract_gallery_faces(image_path):
    image_array = face_recognition.load_image_file(str(image_path))  # Load image as RGB array
    detect_start_time = time.perf_counter()  # Start detection timer
    face_locations = face_recognition.face_locations(
        image_array,
        number_of_times_to_upsample=UPSAMPLE_TIMES,
        model=detection_model,
    )
    detect_seconds = time.perf_counter() - detect_start_time  # Detection elapsed time
    print(f"[face-search] detected {len(face_locations)} face(s) in {format_seconds(detect_seconds)}: {image_path.name}")
    face_records = []  # One record per detected face
    if not face_locations:
        return face_records
    encode_start_time = time.perf_counter()  # Start encoding timer
    face_encodings = face_recognition.face_encodings(
        image_array,
        known_face_locations=face_locations,
        num_jitters=gallery_jitters,
        model=ENCODING_MODEL,
    )
    encode_seconds = time.perf_counter() - encode_start_time  # Encoding elapsed time
    print(f"[face-search] encoded {len(face_encodings)} face(s) in {format_seconds(encode_seconds)}: {image_path.name}")
    face_index = 0  # Index of face inside this image
    for face_encoding in face_encodings:
        raw_embedding = to_raw_embedding(face_encoding)  # Raw dlib embedding
        normalized_embedding = to_normalized_embedding(raw_embedding)  # Unit embedding for cosine
        face_box = face_location_to_box(face_locations[face_index])  # Matched face box
        face_record = {}  # Plain dictionary for simple storage
        face_record["face_index"] = face_index
        face_record["face_box"] = face_box
        face_record["raw_embedding"] = raw_embedding
        face_record["normalized_embedding"] = normalized_embedding
        face_records.append(face_record)
        face_index += 1
    return face_records

def build_faiss_index(database_folder):
    image_paths = list_image_paths(database_folder)  # All gallery images
    raw_embeddings = []  # Raw embeddings for Euclidean display
    normalized_embeddings = []  # Unit embeddings for cosine search
    face_metadata = []  # Metadata aligned with FAISS vector IDs
    scanned_image_count = 0  # Number of images scanned
    skipped_image_count = 0  # Number of images without faces
    failed_image_count = 0  # Number of failed images
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)  # Exact inner-product FAISS index
    print(f"[face-search] found {len(image_paths)} gallery image(s).")
    build_start_time = time.perf_counter()  # Start build timer
    image_number = 1  # Human-readable progress counter
    for image_path in image_paths:
        scanned_image_count += 1
        relative_path = image_path.relative_to(database_folder).as_posix()
        print(f"[face-search] [{image_number}/{len(image_paths)}] indexing: {relative_path}")
        try:
            face_records = extract_gallery_faces(image_path)
            if not face_records:
                skipped_image_count += 1
                print(f"[face-search] no face detected: {relative_path}")
            for face_record in face_records:
                vector_id = len(face_metadata)  # Vector ID equals metadata index
                raw_embeddings.append(face_record["raw_embedding"])
                normalized_embeddings.append(face_record["normalized_embedding"])
                metadata = {}  # Metadata for this face vector
                metadata["vector_id"] = vector_id
                metadata["file_name"] = image_path.name
                metadata["file_path"] = str(image_path)
                metadata["relative_path"] = relative_path
                metadata["face_index"] = face_record["face_index"]
                metadata["face_box"] = face_record["face_box"]
                face_metadata.append(metadata)
        except Exception as error:
            failed_image_count += 1
            print(f"[ERROR] Failed to process {relative_path}: {error}", file=sys.stderr)
        image_number += 1
    raw_embedding_matrix = None  # Raw embedding matrix
    normalized_embedding_matrix = None  # Normalized embedding matrix
    if normalized_embeddings:
        raw_embedding_matrix = np.vstack(raw_embeddings).astype(np.float64)  # Matrix for raw Euclidean scores
        normalized_embedding_matrix = np.vstack(normalized_embeddings).astype(np.float64)  # Matrix for cosine scores
        faiss_matrix = normalized_embedding_matrix.astype(np.float32)  # FAISS uses float32
        faiss_matrix = np.ascontiguousarray(faiss_matrix)  # Contiguous matrix for FAISS
        faiss_index.add(faiss_matrix)  # Add all gallery vectors
    build_seconds = time.perf_counter() - build_start_time  # Build elapsed time
    print()
    print("[face-search] index build finished")
    print(f"  elapsed: {format_seconds(build_seconds)}")
    print(f"  scanned images: {scanned_image_count}")
    print(f"  indexed faces: {len(face_metadata)}")
    print(f"  skipped images without faces: {skipped_image_count}")
    print(f"  failed images: {failed_image_count}")
    print(f"  FAISS ntotal: {faiss_index.ntotal}")
    print()
    return faiss_index, raw_embedding_matrix, normalized_embedding_matrix, face_metadata

def encode_query_image(query_image_path):
    query_image = face_recognition.load_image_file(str(query_image_path))  # Load query image
    detect_start_time = time.perf_counter()  # Start detection timer
    face_locations = face_recognition.face_locations(
        query_image,
        number_of_times_to_upsample=UPSAMPLE_TIMES,
        model=detection_model,
    )
    detection_seconds = time.perf_counter() - detect_start_time  # Detection elapsed time
    if not face_locations:
        raise ValueError("No face detected in query image")
    selected_face_location = face_locations[0]  # Default selected face
    selected_face_area = get_face_area(selected_face_location)  # Area of selected face
    for face_location in face_locations:
        current_face_area = get_face_area(face_location)
        if current_face_area > selected_face_area:
            selected_face_location = face_location
            selected_face_area = current_face_area
    encode_start_time = time.perf_counter()  # Start encoding timer
    face_encodings = face_recognition.face_encodings(
        query_image,
        known_face_locations=[selected_face_location],
        num_jitters=query_jitters,
        model=ENCODING_MODEL,
    )
    encoding_seconds = time.perf_counter() - encode_start_time  # Encoding elapsed time
    if not face_encodings:
        raise ValueError("Failed to extract query face encoding")
    query_raw_embedding = to_raw_embedding(face_encodings[0])  # Raw query embedding
    query_normalized_embedding = to_normalized_embedding(query_raw_embedding)  # Unit query embedding
    selected_face_box = face_location_to_box(selected_face_location)  # Query face box
    return query_raw_embedding, query_normalized_embedding, len(face_locations), selected_face_box, detection_seconds, encoding_seconds

def search_face(query_image_path, faiss_index, raw_embedding_matrix, normalized_embedding_matrix, face_metadata):
    query_data = encode_query_image(query_image_path)  # Encode query face
    query_raw_embedding = query_data[0]
    query_normalized_embedding = query_data[1]
    detected_face_count = query_data[2]
    selected_face_box = query_data[3]
    detection_seconds = query_data[4]
    encoding_seconds = query_data[5]
    indexed_face_count = int(faiss_index.ntotal)  # Number of indexed faces
    result_count = indexed_face_count if TOP_K <= 0 else min(TOP_K, indexed_face_count)  # Number of rows to print
    faiss_query = query_normalized_embedding.astype(np.float32)  # FAISS query vector
    faiss_query = np.ascontiguousarray(faiss_query)  # Contiguous query vector
    faiss_start_time = time.perf_counter()  # Start FAISS timer
    faiss_index.search(faiss_query, indexed_face_count)  # Exact FAISS search for consistency
    faiss_seconds = time.perf_counter() - faiss_start_time  # FAISS elapsed time
    score_start_time = time.perf_counter()  # Start precise scoring timer
    query_unit_vector = query_normalized_embedding.reshape(EMBEDDING_DIMENSION)  # 1D unit query
    cosine_scores = normalized_embedding_matrix @ query_unit_vector  # Float64 cosine scores
    cosine_scores = np.clip(cosine_scores, -1.0, 1.0)  # Clip floating-point overshoot
    query_raw_vector = query_raw_embedding.reshape(EMBEDDING_DIMENSION)  # 1D raw query
    raw_deltas = raw_embedding_matrix - query_raw_vector  # Raw vector differences
    raw_distances = np.linalg.norm(raw_deltas, axis=1)  # Raw Euclidean distances
    normalized_l2_distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine_scores))  # Unit-vector L2
    sorted_indices = np.argsort(-cosine_scores)  # Cosine descending
    results = []  # Final result rows
    rank = 1  # Human-readable rank
    for vector_index in sorted_indices[:result_count]:
        vector_index = int(vector_index)
        metadata = face_metadata[vector_index]
        cosine_similarity = float(cosine_scores[vector_index])
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        similarity_percent = max(0.0, min(100.0, cosine_similarity * 100.0))
        cosine_distance = 1.0 - cosine_similarity
        normalized_l2_distance = float(normalized_l2_distances[vector_index])
        euclidean_distance = float(raw_distances[vector_index])
        result = {}  # Plain dictionary for one result
        result["rank"] = rank
        result["file_name"] = metadata["file_name"]
        result["relative_path"] = metadata["relative_path"]
        result["file_path"] = metadata["file_path"]
        result["face_index"] = metadata["face_index"]
        result["face_box"] = metadata["face_box"]
        result["cosine_similarity"] = cosine_similarity
        result["similarity_percent"] = similarity_percent
        result["cosine_distance"] = cosine_distance
        result["normalized_l2_distance"] = normalized_l2_distance
        result["euclidean_distance"] = euclidean_distance
        result["is_match"] = cosine_similarity >= MATCH_THRESHOLD
        results.append(result)
        rank += 1
    score_seconds = time.perf_counter() - score_start_time  # Precise scoring elapsed time
    stats = {}  # Search statistics
    stats["query_image"] = str(query_image_path)
    stats["detected_face_count"] = detected_face_count
    stats["selected_face_box"] = selected_face_box
    stats["indexed_face_count"] = indexed_face_count
    stats["top_k"] = TOP_K
    stats["returned_count"] = len(results)
    stats["match_count"] = 0
    for result in results:
        if result["is_match"]:
            stats["match_count"] += 1
    stats["detection_seconds"] = detection_seconds
    stats["encoding_seconds"] = encoding_seconds
    stats["faiss_seconds"] = faiss_seconds
    stats["score_seconds"] = score_seconds
    return results, stats

def print_results(results, stats):
    print()
    print("=== SEARCH RESULTS ===")
    print(f"Query image: {stats['query_image']}")
    print(f"Detected faces in query: {stats['detected_face_count']}")
    print(f"Selected query face box: {stats['selected_face_box']}")
    print(f"Indexed faces: {stats['indexed_face_count']}")
    print(f"Top K: {stats['top_k']}")
    print(f"Returned results: {stats['returned_count']}")
    print(f"Match count: {stats['match_count']}")
    print(f"Threshold: cosine_similarity >= {MATCH_THRESHOLD}")
    print("Sort order: cosine_similarity descending")
    print(
        "Timing: "
        f"detect={format_seconds(stats['detection_seconds'])}, "
        f"encode={format_seconds(stats['encoding_seconds'])}, "
        f"faiss={format_seconds(stats['faiss_seconds'])}, "
        f"score={format_seconds(stats['score_seconds'])}"
    )
    print("-" * 96)
    if not results:
        print("No result was returned.")
        print()
        return
    for result in results:
        print(f"Top {result['rank']}")
        print(f"  File name: {result['file_name']}")
        print(f"  Relative path: {result['relative_path']}")
        print(f"  Full file path: {result['file_path']}")
        print(f"  Face index in image: {result['face_index']}")
        print(f"  Face box: {result['face_box']}")
        print(f"  Cosine similarity: {result['cosine_similarity']:.10f}")
        print(f"  Similarity percent: {result['similarity_percent']:.4f}%")
        print(f"  Cosine distance: {result['cosine_distance']:.10f}")
        print(f"  Normalized L2 distance: {result['normalized_l2_distance']:.10f}")
        print(f"  Raw Euclidean distance: {result['euclidean_distance']:.10f}")
        print(f"  Match: {result['is_match']}")
        print("-" * 96)
    print()

def run_search_loop(faiss_index, raw_embedding_matrix, normalized_embedding_matrix, face_metadata):
    print("Enter query image path. Type q, quit, or exit to stop.")
    print()
    while True:
        raw_query_path = input("Enter query image path: ")
        query_path = clean_path(raw_query_path)
        if query_path.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if not query_path:
            print("Query image path cannot be empty.")
            continue
        query_image_path = Path(query_path).expanduser().resolve()
        if not query_image_path.is_file():
            print(f"Invalid query image path: {query_image_path}")
            continue
        try:
            results, stats = search_face(
                query_image_path,
                faiss_index,
                raw_embedding_matrix,
                normalized_embedding_matrix,
                face_metadata,
            )
            print_results(results, stats)
        except Exception as error:
            print(f"[ERROR] Search failed: {error}", file=sys.stderr)

def main():
    database_folder = ask_database_folder()  # Ask once for the gallery folder
    print_runtime_info(database_folder)  # Print selected runtime settings
    index_data = build_faiss_index(database_folder)  # Build the in-memory index
    faiss_index = index_data[0]
    raw_embedding_matrix = index_data[1]
    normalized_embedding_matrix = index_data[2]
    face_metadata = index_data[3]
    if faiss_index.ntotal <= 0:
        print("[ERROR] No faces were indexed. Add face images and run again.", file=sys.stderr)
        return 1
    run_search_loop(faiss_index, raw_embedding_matrix, normalized_embedding_matrix, face_metadata)  # Start query loop
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
