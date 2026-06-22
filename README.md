# 🔍 Face Search Tools

Open-source face search tools for local image datasets.

This repository provides practical tools for face search experiments. The current version includes a single-file Flask application for **1:N face search**. You can upload one query face image, search it against a local image folder, and view ranked similar faces with image previews and matching scores.

Future versions may include more 1:N search tools, N:N comparison, batch indexing, duplicate detection, clustering, API-only services, and other face/vector search utilities.

## ✨ Features

Current tool: `dlib_faiss_flask_1n_single_file.py`

* 🖼️ Local face image database
* 📤 Browser-based query image upload
* 👤 Automatic face detection
* 🧬 Face embedding extraction
* ⚡ FAISS vector search
* 📊 Ranked 1:N search results
* 🖼️ Result image preview
* 📈 Cosine similarity
* 📉 Cosine distance
* 📏 Raw Euclidean distance
* 📐 Normalized L2 distance
* ✅ Match status
* 🎛️ Configurable result count and threshold
* 🧩 Single Python file
* 🌿 Simple Bootstrap interface
* 📜 AGPL-3.0-only license

## 🐧 Linux Setup

Tested-style setup for Ubuntu / Debian.

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3-dev python3-pip
git clone https://github.com/wangyifan349/face-search-tools
cd face-search-tools
python3 -m pip install --upgrade pip setuptools wheel
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
python3 dlib_faiss_flask_1n_single_file.py
```

Open in your browser:

```text
http://127.0.0.1:5000
```

## 🪟 Windows Setup

Install these first:

* **Python 3.9+**
* **Git**
* **CMake**
* **Microsoft Visual Studio Build Tools**

Download Visual Studio Build Tools from Microsoft:

```text
https://visualstudio.microsoft.com/downloads/
```

On the Visual Studio Downloads page, find **Tools for Visual Studio** and download **Build Tools for Visual Studio**.

During installation, select:

```text
Desktop development with C++
```

Make sure these components are installed:

* MSVC C++ build tools
* Windows SDK
* C++ CMake tools for Windows

Then run in PowerShell or CMD:

```powershell
git clone https://github.com/wangyifan349/face-search-tools
cd face-search-tools
python -m pip install --upgrade pip setuptools wheel
pip install numpy pillow flask faiss-cpu
pip install dlib
pip install face_recognition
python dlib_faiss_flask_1n_single_file.py
```

Open in your browser:

```text
http://127.0.0.1:5000
```

If `dlib` fails to install on Windows, check that Visual Studio Build Tools, CMake, and the C++ build environment are installed correctly. After installing or updating Build Tools, restart PowerShell or CMD and run the `pip install dlib` command again.

## 📋 Result Fields

| Field                  | Meaning                                            |
| ---------------------- | -------------------------------------------------- |
| Rank                   | Search result order                                |
| Image                  | Preview of the matched image                       |
| File path              | Local source image path                            |
| Cosine similarity      | Higher means more similar                          |
| Cosine distance        | Lower means more similar                           |
| Raw Euclidean distance | Lower means closer raw face embeddings             |
| Normalized L2 distance | Lower means closer normalized embeddings           |
| Match status           | Whether the result passes the configured threshold |
| Open / Download        | Browser links for the result image                 |

## ⚙️ Configuration

You can adjust these values inside the Python file:

* Image database folder
* Upload folder
* Server host
* Server port
* Default top-k result count
* Match threshold
* Face encoding jitter count
* Supported image extensions

## ☕ Buy Me a Coffee

If this project helps you, you can optionally support the author. Thank you for your kindness.

### ₿ Bitcoin

```text
bc1qxqfhumpqtnxrznkx9r4xsp8m6zsedtgusjns7p
```

### Ξ Ethereum

```text
0x2d92f9e4d8ac7effa9cd7cd5eccd364cac7c201b
```

### ◎ Solana

```text
B7N4e3KG9zWQBwMrtydS1B9wVBp2w62fAdryZdxAMBiz
```

### 💵 USDT Ethereum / ERC-20

```text
0x2d92f9e4d8ac7effa9cd7cd5eccd364cac7c201b
```

## 🙏 Acknowledgements

Special thanks to the excellent open-source project:

```text
https://github.com/ageitgey/face_recognition
```

This repository uses `face_recognition` / dlib-style face embeddings as an important foundation for face detection and face encoding.

Thanks also to the open-source communities behind Flask, FAISS, NumPy, Pillow, dlib, and related Python tooling.

## 🔒 Privacy

Face images and face embeddings may be sensitive biometric data.

Make sure you have the legal right and user consent to process the images. Do not publish private face datasets in a public repository.

Do not commit private images, uploaded files, generated indexes, embeddings, or runtime data.

## 🛣️ Future Plans

Possible future additions:

* More 1:N face search tools
* N:N face comparison
* Batch face indexing
* Duplicate face detection
* Face clustering
* Persistent FAISS indexes
* API-only search service
* Multi-model embedding comparison
* Larger dataset search demos

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
