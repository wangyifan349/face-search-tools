# 🔍 Face Search Tools

Practical open-source face search tools for local image datasets.

This repository starts with a simple single-file Flask application for **1:N face search**. You upload one query face image, search it against a local face image folder, and view ranked similar faces with image previews and useful matching scores.

Future tools may include more 1:N search utilities, N:N comparison, batch indexing, duplicate detection, clustering, API-only services, and other face/vector search experiments.

## ✨ Features

Current tool: `dlib_faiss_flask_1n_single_file.py`

- 🖼️ Local face image database
- 📤 Browser-based query image upload
- 👤 Automatic face detection
- 🧬 128-dimensional face embedding extraction
- ⚡ FAISS vector search
- 📊 Ranked 1:N search results
- 🖼️ Result image preview
- 📈 Cosine similarity display
- 📉 Cosine distance display
- 📏 Raw Euclidean distance display
- 📐 Normalized L2 distance display
- ✅ Match status based on configurable threshold
- 🎛️ Configurable result count
- 🧩 Single Python file for easier testing and deployment
- 🌿 Simple Bootstrap interface with a clean visual style
- 📜 AGPL-3.0 open-source license

## 📦 Repository

```text
https://github.com/wangyifan349/face-search-tools
```

Suggested GitHub repository description:

```text
Simple face search tools using dlib face embeddings, FAISS, cosine similarity, and Euclidean distance.
```

## 🧰 Environment Requirements

Recommended:

- Python 3.9+
- pip
- CMake
- C++ build tools
- dlib
- face_recognition
- FAISS CPU
- Flask
- NumPy
- Pillow

The most important dependency is `dlib`. The Python package `face_recognition` depends on `dlib`, so the system must be able to compile or install dlib correctly.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/wangyifan349/face-search-tools.git
cd face-search-tools
```

### 2. Create a virtual environment

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 🐧 Linux Deployment

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-dev python3-pip python3-venv
```

Then install Python packages:

```bash
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
```

Or install them together:

```bash
pip install numpy pillow flask faiss-cpu dlib face_recognition
```

If `dlib` installation fails, check that `cmake`, a C++ compiler, and Python development headers are installed.

## 🍎 macOS Deployment

Install build tools:

```bash
xcode-select --install
```

If you use Homebrew:

```bash
brew install cmake
```

Then install Python packages:

```bash
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
```

## 🪟 Windows Deployment

Windows dlib installation may require Visual Studio Build Tools.

Install:

- Python 3.9+
- CMake
- Visual Studio Build Tools with C++ build tools

Then run:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
```

If building `dlib` fails on Windows, using Conda is often easier:

```powershell
conda create -n face-search python=3.9 -y
conda activate face-search
conda install -c conda-forge dlib -y
pip install face_recognition flask numpy pillow faiss-cpu
```

## 🧪 Quick Start

Put gallery images into the configured local image folder.

By default, the current tool is designed for folders such as:

```text
face_database/
runtime_uploads/
```

Run the application:

```bash
python dlib_faiss_flask_1n_single_file.py
```

Open in your browser:

```text
http://127.0.0.1:5000
```

Upload a query image and view ranked search results.

## 🗂️ Suggested Project Structure

```text
face-search-tools/
├── README.md
├── LICENSE
├── .gitignore
├── dlib_faiss_flask_1n_single_file.py
├── face_database/
└── runtime_uploads/
```

Do not commit private images, upload files, generated indexes, embeddings, or runtime data.

## 📋 Result Fields

Each result displays practical information:

| Field | Meaning |
| --- | --- |
| Rank | Search result order |
| Image | Preview of the matched image |
| File path | Local source image path |
| Cosine similarity | Higher means more similar |
| Cosine distance | Lower means more similar |
| Raw Euclidean distance | Lower means closer raw face embeddings |
| Normalized L2 distance | Lower means closer normalized embeddings |
| Match status | Whether the result passes the configured threshold |
| Open / Download | Browser links for the result image |

## ⚙️ Configuration

The current tool is intentionally kept as a single Python file.

Common items you may want to adjust inside the script:

- Image database folder
- Upload folder
- Server host
- Server port
- Default top-k result count
- Match threshold
- Face encoding jitter count
- Supported image extensions

## 🧠 Notes

This project is for research, testing, demos, and engineering experiments.

Face search quality depends on:

- Image quality
- Lighting
- Face angle
- Occlusion
- Resolution
- Dataset quality
- Face detection behavior
- Threshold settings

The default threshold is only a practical starting point. For real use, evaluate and calibrate thresholds with your own dataset.

## 🔒 Privacy

Face images and face embeddings may be sensitive biometric data.

Before using this project, make sure you have the legal right and user consent to process the images.

Do not publish private face datasets in a public repository.

Recommended ignored files are already included in `.gitignore`.

## 🛣️ Future Plans

Possible future additions:

- More 1:N face search tools
- N:N face comparison
- Batch face indexing
- Duplicate face detection
- Face clustering
- Persistent FAISS indexes
- API-only search service
- Multi-model embedding comparison
- Larger dataset search demos

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 only**.

SPDX-License-Identifier:

```text
AGPL-3.0-only
```

See the `LICENSE` file for details.

## ⚠️ Disclaimer

This repository is provided for research, learning, testing, and engineering experiments only.

The author is not responsible for misuse, privacy issues, incorrect matching results, or decisions made based on this software.
