#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InsightFace 1:N face similarity search.
Usage:
    python insightface_search.py ./gallery ./test.jpg
Windows CPU:
    python -m pip install --upgrade pip
    pip install insightface opencv-python numpy onnxruntime
Windows GPU:
    python -m pip install --upgrade pip
    pip install insightface opencv-python numpy onnxruntime-gpu
    Make sure NVIDIA Driver, CUDA and cuDNN versions match your onnxruntime-gpu version.
Linux CPU:
    python3 -m pip install --upgrade pip
    pip install insightface opencv-python numpy onnxruntime
Linux GPU:
    python3 -m pip install --upgrade pip
    pip install insightface opencv-python numpy onnxruntime-gpu
    Make sure NVIDIA Driver, CUDA and cuDNN versions match your onnxruntime-gpu version.
Gallery example:
    gallery/person_a/001.jpg
    gallery/person_b/001.jpg
Notes:
    - Parent directory name is used as the person name.
    - CUDA is detected automatically through ONNX Runtime providers.
    - CUDA mode uses larger detection size for better face detection.
    - CPU mode uses smaller detection size for better compatibility.
    - Search uses cosine similarity on normalized InsightFace embeddings.
"""
import argparse
import os
from typing import Any
import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

FACE_ANALYSIS_MODEL_NAME = "buffalo_l"
MATCH_THRESHOLD = 0.35
CPU_DETECTION_SIZE = (640, 640)
CUDA_DETECTION_SIZE = (1024, 1024)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
def get_available_execution_providers() -> list[str]:
    """Return all ONNX Runtime execution providers available in the current environment."""
    return onnxruntime.get_available_providers()
def is_cuda_execution_available() -> bool:
    """Return True if ONNX Runtime can use CUDAExecutionProvider."""
    return "CUDAExecutionProvider" in get_available_execution_providers()
def get_runtime_configuration() -> dict[str, Any]:
    """Select execution providers, context identifier, and detection size."""
    if is_cuda_execution_available():
        return {
            "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "context_identifier": 0,
            "detection_size": CUDA_DETECTION_SIZE,
            "runtime_device": "CUDA",
        }
    return {
        "execution_providers": ["CPUExecutionProvider"],
        "context_identifier": -1,
        "detection_size": CPU_DETECTION_SIZE,
        "runtime_device": "CPU",
    }

def load_face_analysis_application(runtime_configuration: dict[str, Any]) -> FaceAnalysis:
    """Load InsightFace FaceAnalysis with the selected model and runtime configuration."""
    face_analysis_application = FaceAnalysis(name=FACE_ANALYSIS_MODEL_NAME, providers=runtime_configuration["execution_providers"])
    face_analysis_application.prepare(ctx_id=runtime_configuration["context_identifier"], det_size=runtime_configuration["detection_size"])
    return face_analysis_application

def collect_image_paths(gallery_directory: str) -> list[str]:
    """Recursively collect supported image paths from the gallery directory."""
    image_paths = []
    for root_directory, _, file_names in os.walk(gallery_directory):
        for file_name in file_names:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root_directory, file_name))
    return image_paths

def calculate_face_area(face: Any) -> float:
    """Calculate face bounding box area."""
    bounding_box = face.bbox
    return float((bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1]))
def extract_face_embedding(face_analysis_application: FaceAnalysis, image_path: str) -> np.ndarray:
    """Extract normalized InsightFace embedding from the largest detected face."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("failed to read image")
    detected_faces = face_analysis_application.get(image)
    if not detected_faces:
        raise ValueError("no face detected")
    largest_face = max(detected_faces, key=calculate_face_area)
    return np.asarray(largest_face.normed_embedding, dtype=np.float32)

def build_face_gallery(face_analysis_application: FaceAnalysis, gallery_directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build gallery embeddings. The parent directory name is used as the person name."""
    image_paths = collect_image_paths(gallery_directory)
    person_names, face_image_paths, face_embeddings = [], [], []
    successful_face_count, failed_image_count = 0, 0
    for image_index, image_path in enumerate(image_paths, 1):
        try:
            face_embedding = extract_face_embedding(face_analysis_application, image_path)
            person_names.append(os.path.basename(os.path.dirname(image_path)))
            face_image_paths.append(image_path)
            face_embeddings.append(face_embedding)
            successful_face_count += 1
            print(f"[OK] {image_index}/{len(image_paths)} {image_path}")
        except Exception as error:
            failed_image_count += 1
            print(f"[SKIP] {image_index}/{len(image_paths)} {image_path} -> {error}")
    if not face_embeddings:
        raise RuntimeError("No face embeddings were extracted from gallery")
    gallery_statistics = {
        "total_image_count": len(image_paths),
        "successful_face_count": successful_face_count,
        "failed_image_count": failed_image_count,
    }
    return np.array(person_names), np.array(face_image_paths), np.vstack(face_embeddings).astype(np.float32), gallery_statistics

def search_face_gallery(query_embedding: np.ndarray, gallery_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Search the gallery by cosine similarity. The embeddings are already normalized."""
    cosine_similarities = np.dot(gallery_embeddings, query_embedding)
    sorted_indices = np.argsort(-cosine_similarities)
    return cosine_similarities, sorted_indices

def print_environment_report(runtime_configuration: dict[str, Any]) -> None:
    """Print runtime and model configuration."""
    print("Runtime")
    print("-------")
    print(f"Model Name          : {FACE_ANALYSIS_MODEL_NAME}")
    print(f"Runtime Device      : {runtime_configuration['runtime_device']}")
    print(f"CUDA Available      : {is_cuda_execution_available()}")
    print(f"Detection Size      : {runtime_configuration['detection_size']}")
    print(f"Selected Providers  : {runtime_configuration['execution_providers']}")
    print(f"Available Providers : {get_available_execution_providers()}")

def print_search_report(
    query_image_path: str,
    runtime_configuration: dict[str, Any],
    person_names: np.ndarray,
    face_image_paths: np.ndarray,
    cosine_similarities: np.ndarray,
    sorted_indices: np.ndarray,
    gallery_statistics: dict[str, int],
) -> None:
    """Print a clear 1:N search report."""
    matched_indices = [int(index) for index in sorted_indices if float(cosine_similarities[int(index)]) >= MATCH_THRESHOLD]
    print("\n==================================================")
    print("InsightFace 1:N Face Search")
    print("==================================================")
    print(f"Query Image         : {query_image_path}")
    print(f"Match Rule          : cosine similarity >= {MATCH_THRESHOLD}")
    print()
    print_environment_report(runtime_configuration)
    print("\nGallery")
    print("-------")
    print(f"Total Images        : {gallery_statistics['total_image_count']}")
    print(f"Success Faces       : {gallery_statistics['successful_face_count']}")
    print(f"Failed Images       : {gallery_statistics['failed_image_count']}")
    print(f"Gallery Faces       : {len(person_names)}")
    print("\nResults")
    print("-------")
    print(f"Matched Faces       : {len(matched_indices)}")
    if matched_indices:
        for rank, index in enumerate(matched_indices, 1):
            cosine_similarity = float(cosine_similarities[index])
            similarity_percentage = cosine_similarity * 100
            print(f"\nRank                : #{rank}")
            print(f"Person              : {person_names[index]}")
            print(f"Cosine Similarity   : {cosine_similarity:.6f}")
            print(f"Similarity          : {similarity_percentage:.2f}%")
            print(f"Image               : {face_image_paths[index]}")
    else:
        closest_index = int(sorted_indices[0])
        closest_cosine_similarity = float(cosine_similarities[closest_index])
        print("\nClosest Candidate")
        print("-----------------")
        print(f"Person              : {person_names[closest_index]}")
        print(f"Cosine Similarity   : {closest_cosine_similarity:.6f}")
        print(f"Similarity          : {closest_cosine_similarity * 100:.2f}%")
        print(f"Image               : {face_image_paths[closest_index]}")
        print(f"Reason              : cosine similarity < {MATCH_THRESHOLD}")
    print("\n==================================================")

def parse_command_line_arguments() -> argparse.Namespace:
    """Parse positional command line arguments."""
    argument_parser = argparse.ArgumentParser(description="InsightFace 1:N face similarity search")
    argument_parser.add_argument("gallery", help="gallery root directory")
    argument_parser.add_argument("query", help="query image path")
    return argument_parser.parse_args()

def main() -> None:
    command_line_arguments = parse_command_line_arguments()
    runtime_configuration = get_runtime_configuration()
    face_analysis_application = load_face_analysis_application(runtime_configuration)
    person_names, face_image_paths, gallery_embeddings, gallery_statistics = build_face_gallery(face_analysis_application, command_line_arguments.gallery)
    query_embedding = extract_face_embedding(face_analysis_application, command_line_arguments.query)
    cosine_similarities, sorted_indices = search_face_gallery(query_embedding, gallery_embeddings)
    print_search_report(
        command_line_arguments.query,
        runtime_configuration,
        person_names,
        face_image_paths,
        cosine_similarities,
        sorted_indices,
        gallery_statistics,
    )
if __name__ == "__main__":
    main()
