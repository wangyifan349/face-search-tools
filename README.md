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
- 🌿 Simple Bootstrap interface
- 📜 AGPL-3.0 open-source license

## 📦 Repository

```text
https://github.com/wangyifan349/face-search-tools
```

Suggested GitHub description:

```text
Open-source face search toolkit for local image datasets, including 1:N search with dlib embeddings, FAISS ranking, cosine similarity, and Euclidean distance.
```

## 🧰 Requirements

Recommended environment:

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

`face_recognition` depends on `dlib`. If `dlib` installation fails, make sure CMake, a C++ compiler, and Python development headers are installed.

## 🚀 Quick Start

```bash
git clone https://github.com/wangyifan349/face-search-tools
cd face-search-tools
python -m pip install --upgrade pip setuptools wheel
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
python dlib_faiss_flask_1n_single_file.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 🐧 Linux Setup

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-dev python3-pip python3-venv
git clone https://github.com/wangyifan349/face-search-tools
cd face-search-tools
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
python dlib_faiss_flask_1n_single_file.py
```

Open:

```text
http://127.0.0.1:5000
```

## 🪟 Windows Setup

Install these first:

- Python 3.9+
- CMake
- Visual Studio Build Tools with C++ build tools
- Git

Then run in PowerShell:

```powershell
git clone https://github.com/wangyifan349/face-search-tools
cd face-search-tools
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
python dlib_faiss_flask_1n_single_file.py
```

Open:

```text
http://127.0.0.1:5000
```

If `dlib` fails to build on Windows, install Visual Studio Build Tools and CMake first, then try again.

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

The tool is intentionally kept as a single Python file.

Common items you may want to adjust inside the script:

- Image database folder
- Upload folder
- Server host
- Server port
- Default top-k result count
- Match threshold
- Face encoding jitter count
- Supported image extensions

## ☕ Buy Me a Coffee

If this project helps you, you can optionally support the author.

| Coin | Address |
| --- | --- |
| BTC | `bc1qxqfhumpqtnxrznkx9r4xsp8m6zsedtgusjns7p` |
| ETH | `0x2d92f9e4d8ac7effa9cd7cd5eccd364cac7c201b` |
| SOL | `B7N4e3KG9zWQBwMrtydS1B9wVBp2w62fAdryZdxAMBiz` |
| USDT Ethereum / ERC-20 | `0x2d92f9e4d8ac7effa9cd7cd5eccd364cac7c201b` |

Thank you for your support.

## 🧠 Notes

This project is for research, testing, demos, and engineering experiments.

Face search quality depends on image quality, lighting, face angle, occlusion, resolution, dataset quality, face detection behavior, and threshold settings.

The default threshold is only a practical starting point. For real use, evaluate and calibrate thresholds with your own dataset.

## 🔒 Privacy

Face images and face embeddings may be sensitive biometric data.

Before using this project, make sure you have the legal right and user consent to process the images.

Do not publish private face datasets in a public repository.

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
