"""
InsightFace Face Analysis Script.

This script automatically selects CUDA when available, otherwise falls back to CPU.
It detects faces, extracts face attributes, prints analysis details, calculates
all-to-all face similarity, draws face annotations, and saves the output image.
"""

import logging
from typing import Any, List

import cv2
import insightface
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.data import get_image


MIN_INSIGHTFACE_VERSION = "0.3"
SAMPLE_IMAGE_NAME = "t1"
OUTPUT_IMAGE_PATH = "./face_analysis_result.jpg"
DETECTION_SIZE = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_context_id() -> int:
    """Return CUDA context id when CUDA is available, otherwise return CPU id."""
    available_providers = ort.get_available_providers()  # Read ONNX Runtime providers

    logging.info("Available ONNX Runtime providers:")
    for provider_name in available_providers:
        logging.info("  %s", provider_name)

    if "CUDAExecutionProvider" in available_providers:
        logging.info("Execution device: CUDA")
        return 0

    logging.info("Execution device: CPU")
    return -1


def create_face_app(context_id: int) -> FaceAnalysis:
    """Create and prepare the InsightFace application."""
    face_app = FaceAnalysis()  # Initialize InsightFace application
    face_app.prepare(ctx_id=context_id, det_size=DETECTION_SIZE)  # Load models
    return face_app


def get_gender_name(gender_value: int) -> str:
    """Convert InsightFace gender value to readable text."""
    if gender_value == 0:
        return "Female"

    if gender_value == 1:
        return "Male"

    return "Unknown"


def print_face_info(face_index: int, face: Any) -> None:
    """Print all available information for a single face."""
    available_fields = face.keys()  # Read available fields from face object

    logging.info("Face %s", face_index + 1)
    logging.info("Fields: %s", available_fields)

    if "bbox" in available_fields:
        logging.info("Bounding box: %s", face.bbox)

    if "det_score" in available_fields:
        logging.info("Detection score: %s", face.det_score)

    if "age" in available_fields:
        logging.info("Age: %s", face.age)

    if "gender" in available_fields:
        gender_name = get_gender_name(face.gender)
        logging.info("Gender: %s", gender_name)

    if "pose" in available_fields:
        logging.info("Head pose yaw/pitch/roll: %s", face.pose)

    if "kps" in available_fields:
        logging.info("5-point landmarks: %s", face.kps)

    if "landmark_2d_106" in available_fields:
        logging.info("106-point landmarks shape: %s", face.landmark_2d_106.shape)

    if "embedding" in available_fields:
        logging.info("Embedding shape: %s", face.embedding.shape)

    if "normed_embedding" in available_fields:
        logging.info("Normalized embedding shape: %s", face.normed_embedding.shape)

    if "score" in available_fields:
        logging.info("Quality score: %s", face.score)


def collect_normalized_embeddings(faces: List[Any]) -> np.ndarray:
    """Collect normalized face embeddings into a NumPy matrix."""
    embedding_list = []  # Store normalized embeddings

    for face in faces:
        available_fields = face.keys()

        if "normed_embedding" not in available_fields:
            continue

        embedding_list.append(face.normed_embedding)

    if len(embedding_list) == 0:
        return np.empty((0, 0), dtype=np.float32)

    embedding_matrix = np.array(embedding_list, dtype=np.float32)
    return embedding_matrix


def calculate_similarity_matrix(embedding_matrix: np.ndarray) -> np.ndarray:
    """Calculate all-to-all cosine similarity for normalized embeddings."""
    if embedding_matrix.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
    return similarity_matrix


def save_annotated_image(face_app: FaceAnalysis, image: np.ndarray, faces: List[Any]) -> None:
    """Draw face analysis results and save the annotated image."""
    result_image = face_app.draw_on(image, faces)  # Draw boxes and landmarks
    save_success = cv2.imwrite(OUTPUT_IMAGE_PATH, result_image)  # Save output image

    if not save_success:
        raise RuntimeError(f"Failed to save output image: {OUTPUT_IMAGE_PATH}")

    logging.info("Saved output image: %s", OUTPUT_IMAGE_PATH)


def main() -> None:
    """Run the InsightFace face analysis workflow."""
    assert insightface.__version__ >= MIN_INSIGHTFACE_VERSION

    context_id = get_context_id()
    face_app = create_face_app(context_id)

    image = get_image(SAMPLE_IMAGE_NAME)  # Load sample image
    faces = face_app.get(image)  # Detect and analyze faces

    logging.info("Detected faces: %s", len(faces))

    for face_index, face in enumerate(faces):
        print_face_info(face_index, face)

    embedding_matrix = collect_normalized_embeddings(faces)
    similarity_matrix = calculate_similarity_matrix(embedding_matrix)

    if similarity_matrix.size > 0:
        np.set_printoptions(precision=4, suppress=True)
        logging.info("Face similarity matrix:\n%s", similarity_matrix)

    save_annotated_image(face_app, image, faces)


if __name__ == "__main__":
    main()
