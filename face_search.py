#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure dlib face similarity 1:N search.
Usage:
    python face_search.py ./gallery ./test.jpg
Install:
    pip install numpy dlib
If dlib installation fails on Ubuntu/Debian:
    sudo apt update
    sudo apt install -y build-essential cmake python3-dev
    pip install numpy dlib
Conda:
    conda install -c conda-forge dlib numpy
Required model files, placed next to this script:
    shape_predictor_68_face_landmarks.dat
    dlib_face_recognition_resnet_model_v1.dat
    mmod_human_face_detector.dat
Behavior:
    - Recursively scans gallery with os.walk().
    - Uses parent directory name as person name.
    - Uses CUDA + CNN detector if available, otherwise HOG detector.
    - Extracts dlib 128D face embeddings.
    - Compares query image with all gallery embeddings by Euclidean distance.
    - Prints all matches with distance <= 0.6, sorted from most similar to least similar.
    - No cache, no timing statistics.
"""
import argparse
import os
from pathlib import Path
import dlib
import numpy as np

MATCH_THRESHOLD = 0.6
SCRIPT_DIR = Path(__file__).resolve().parent
LANDMARK_MODEL_PATH = SCRIPT_DIR / "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = SCRIPT_DIR / "dlib_face_recognition_resnet_model_v1.dat"
CNN_FACE_DETECTOR_MODEL_PATH = SCRIPT_DIR / "mmod_human_face_detector.dat"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def is_cuda_available():
    """Return True only when dlib is compiled with CUDA and at least one CUDA device is visible."""
    try:
        return bool(dlib.DLIB_USE_CUDA) and dlib.cuda.get_num_devices() > 0
    except Exception:
        return False

def get_cuda_device_count():
    """Return CUDA device count reported by dlib."""
    try:
        return dlib.cuda.get_num_devices()
    except Exception:
        return 0

def validate_model_files():
    """Check required model files before running."""
    if not LANDMARK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing landmark model: {LANDMARK_MODEL_PATH}")
    if not FACE_RECOGNITION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing face recognition model: {FACE_RECOGNITION_MODEL_PATH}")

def print_environment_report():
    """Print dlib and CUDA environment information."""
    print("Environment")
    print("-----------")
    print(f"dlib Version      : {dlib.__version__}")
    print(f"DLIB_USE_CUDA     : {bool(dlib.DLIB_USE_CUDA)}")
    print(f"CUDA Devices      : {get_cuda_device_count()}")
    print(f"CUDA Available    : {is_cuda_available()}")

def load_face_detector():
    """Use CNN detector when CUDA is available; otherwise use HOG detector."""
    if is_cuda_available():
        if not CNN_FACE_DETECTOR_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing CNN face detector model: {CNN_FACE_DETECTOR_MODEL_PATH}")
        print("[INFO] Using CNN face detector")
        return "cnn", dlib.cnn_face_detection_model_v1(str(CNN_FACE_DETECTOR_MODEL_PATH))
    print("[INFO] Using HOG face detector")
    return "hog", dlib.get_frontal_face_detector()

def detect_faces(image_rgb, detector_mode, face_detector):
    """Detect face rectangles. CNN detector returns mmod rectangles, HOG returns rectangles directly."""
    if detector_mode == "cnn":
        return [detection.rect for detection in face_detector(image_rgb, 1)]
    return list(face_detector(image_rgb, 1))

def collect_image_paths(gallery_directory):
    """Recursively collect image paths from gallery directory."""
    image_paths = []
    for root_directory, _, file_names in os.walk(gallery_directory):
        for file_name in file_names:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root_directory, file_name))
    return image_paths

def extract_face_embedding(image_path, detector_mode, face_detector, landmark_model, face_recognition_model):
    """Extract one 128D dlib face embedding from the largest detected face in an image."""
    image_rgb = dlib.load_rgb_image(image_path)
    face_rectangles = detect_faces(image_rgb, detector_mode, face_detector)
    if not face_rectangles:
        raise ValueError("no face detected")
    largest_face_rectangle = max(face_rectangles, key=lambda rectangle: rectangle.width() * rectangle.height())
    face_landmarks = landmark_model(image_rgb, largest_face_rectangle)
    face_descriptor = face_recognition_model.compute_face_descriptor(image_rgb, face_landmarks)
    return np.asarray(face_descriptor, dtype=np.float32)

def build_face_gallery(gallery_directory, detector_mode, face_detector, landmark_model, face_recognition_model):
    """Build gallery embeddings. Parent directory name is used as person name."""
    image_paths = collect_image_paths(gallery_directory)
    person_names, face_image_paths, face_embeddings = [], [], []
    successful_face_count, failed_image_count = 0, 0
    for image_index, image_path in enumerate(image_paths, 1):
        try:
            face_embedding = extract_face_embedding(
                image_path, detector_mode, face_detector, landmark_model, face_recognition_model
            )
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

def search_face_gallery(query_embedding, gallery_embeddings):
    """Linear 1:N search by Euclidean distance."""
    face_distances = np.linalg.norm(gallery_embeddings - query_embedding, axis=1)
    sorted_indices = np.argsort(face_distances)
    return face_distances, sorted_indices

def print_search_report(query_image_path, detector_mode, person_names, face_image_paths, face_distances, sorted_indices, gallery_statistics):
    """Print environment, gallery summary, and all valid matches."""
    matched_indices = [int(index) for index in sorted_indices if float(face_distances[int(index)]) <= MATCH_THRESHOLD]
    print("\n==================================================")
    print("dlib Face Search")
    print("==================================================")
    print(f"Query Image       : {query_image_path}")
    print(f"Detector          : {detector_mode.upper()}")
    print(f"Threshold         : {MATCH_THRESHOLD}\n")
    print_environment_report()
    print("\nModels")
    print("------")
    print(f"Landmark Model    : {LANDMARK_MODEL_PATH.name}")
    print(f"Recognition Model : {FACE_RECOGNITION_MODEL_PATH.name}")
    if detector_mode == "cnn":
        print(f"CNN Detector      : {CNN_FACE_DETECTOR_MODEL_PATH.name}")
    print("\nGallery Statistics")
    print("------------------")
    print(f"Total Images      : {gallery_statistics['total_image_count']}")
    print(f"Success Faces     : {gallery_statistics['successful_face_count']}")
    print(f"Failed Images     : {gallery_statistics['failed_image_count']}")
    print(f"Gallery Faces     : {len(person_names)}")
    print("\nMatches")
    print("-------")
    print(f"Found             : {len(matched_indices)}")
    if matched_indices:
        for rank, index in enumerate(matched_indices, 1):
            print(f"\n#{rank}")
            print(f"Person            : {person_names[index]}")
            print(f"Distance          : {float(face_distances[index]):.6f}")
            print(f"File              : {face_image_paths[index]}")
    else:
        closest_index = int(sorted_indices[0])
        print("\nClosest Candidate")
        print("-----------------")
        print(f"Person            : {person_names[closest_index]}")
        print(f"Distance          : {float(face_distances[closest_index]):.6f}")
        print(f"File              : {face_image_paths[closest_index]}")
        print(f"Reason            : distance > {MATCH_THRESHOLD}")
    print("\n==================================================")

def parse_arguments():
    """Parse positional CLI arguments: gallery directory and query image."""
    argument_parser = argparse.ArgumentParser(description="Pure dlib face similarity 1:N search")
    argument_parser.add_argument("gallery", help="gallery root directory")
    argument_parser.add_argument("query", help="query image path")
    return argument_parser.parse_args()

def main():
    arguments = parse_arguments()
    validate_model_files()
    detector_mode, face_detector = load_face_detector()
    landmark_model = dlib.shape_predictor(str(LANDMARK_MODEL_PATH))
    face_recognition_model = dlib.face_recognition_model_v1(str(FACE_RECOGNITION_MODEL_PATH))
    person_names, face_image_paths, gallery_embeddings, gallery_statistics = build_face_gallery(
        arguments.gallery, detector_mode, face_detector, landmark_model, face_recognition_model
    )
    query_embedding = extract_face_embedding(
        arguments.query, detector_mode, face_detector, landmark_model, face_recognition_model
    )
    face_distances, sorted_indices = search_face_gallery(query_embedding, gallery_embeddings)
    print_search_report(
        arguments.query,
        detector_mode,
        person_names,
        face_image_paths,
        face_distances,
        sorted_indices,
        gallery_statistics,
    )

if __name__ == "__main__":
    main()
