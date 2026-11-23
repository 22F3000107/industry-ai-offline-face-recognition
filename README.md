# 👤 Who Is in the Photo?

### Privacy-First Offline Face Recognition using OpenCV YuNet and SFace

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%7C%20SFace-green)](https://github.com/opencv/opencv_zoo)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![Processing](https://img.shields.io/badge/Processing-100%25%20Offline-success)]()

An offline face recognition application that identifies people in group photographs by comparing detected faces against a locally maintained reference database.

The system uses **OpenCV YuNet** for face detection and **OpenCV SFace** for face feature extraction and recognition.

Photographs and face embeddings are processed locally. No cloud-based face recognition services or online recognition APIs are used.

---

## 🚀 Project Overview

Who Is in the Photo? is a privacy-first face recognition system designed to:

- Enroll individuals using their reference photographs.
- Detect multiple faces in a group photograph.
- Generate facial embeddings locally.
- Compare detected faces against enrolled identities.
- Identify known individuals.
- Label unmatched individuals as **Unknown**.
- Display annotated photographs and recognition results.

The application is built using Python, OpenCV and Streamlit, with pretrained ONNX models for face detection and recognition.

## ✨ Key Features

### 1. Reference Face Enrollment
- Select a local folder containing individually named photographs.
- Detect faces in each reference image.
- Require exactly one detectable face per reference photograph.
- Generate and store face embeddings.
- Save and load the local embedding database.

### 2. Group Photograph Recognition
- Upload a group photograph.
- Detect multiple faces automatically.
- Generate embeddings for detected faces.
- Compare embeddings against enrolled identities.
- Identify known people or return `Unknown`.

### 3. Face Annotation
- Draw bounding boxes around detected faces.
- Display identity labels and similarity scores.
- Use different visual indicators for recognized and unknown individuals.

### 4. Privacy-First Offline Processing
- Local image processing.
- Local ONNX model inference.
- Local embedding database.
- No online face recognition APIs.
- No cloud-based image processing.

## 🏗️ System Architecture

The application follows this processing pipeline:

```text
                USER
                 |
          Streamlit Interface
                 |
          +------+------+
          |             |
    Reference Photos   Group Photo
          |             |
          v             v
       YuNet Face Detection
          |             |
          v             v
       SFace Embedding Extraction
          |             |
          v             |
    Reference Database  |
          |             |
          +------v------+
             Face Matcher
                  |
          Cosine Similarity
                  |
          Threshold Check
                  |
          +-------+-------+
          |               |
       Recognized       Unknown
          |               |
          +-------+-------+
                  |
          Annotated Image
                  |
          Recognition Results
```

### Recognition Decision

The system compares normalized face embeddings using cosine similarity.

\[
Similarity(q,r)=q\cdot r
\]

Where:

- \(q\) represents the query face embedding.
- \(r\) represents a reference face embedding.

The initial recognition threshold is:

```python
threshold = 0.363
```

If the highest similarity score meets or exceeds the threshold, the corresponding identity is returned. Otherwise, the face is classified as `Unknown`.

The threshold is configurable and should be calibrated using representative validation data.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Face detection | OpenCV YuNet |
| Face recognition | OpenCV SFace |
| Model format | ONNX |
| Image processing | OpenCV |
| Numerical operations | NumPy |
| Data handling | Pandas |
| Image handling | Pillow |
| Testing | Python unittest / custom validation scripts |
| Version control | Git and GitHub |

## 📁 Project Structure

```text
industry-ai-offline-face-recognition/
│
├── app.py
├── requirements.txt
├── MODEL_LICENSES.md
├── README.md
├── .gitignore
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── face_detector.py
│   ├── face_encoder.py
│   └── face_matcher.py
│
└── tests/
    ├── __init__.py
    ├── check_lfw_references.py
    ├── complete_50_references.py
    ├── test_20_unknown_validation.py
    ├── test_50_person_validation.py
    ├── test_confidence_margin.py
    ├── test_database_manual.py
    ├── test_face_detection_manual.py
    ├── test_face_encoding_manual.py
    ├── test_face_matching_manual.py
    ├── test_lfw_validation.py
    ├── test_recognition_validation.py
    └── test_unknown_rejection.py
```

Local datasets, reference photographs, generated embeddings and validation outputs are excluded from Git.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- A local computer capable of running OpenCV inference.
- macOS, Linux or Windows.

### 1. Clone the repository

```bash
git clone https://github.com/22F3000107/industry-ai-offline-face-recognition.git

cd industry-ai-offline-face-recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify model files

Ensure the following pretrained ONNX files exist:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

Both models are included in this repository.

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local application in your browser:

```text
http://localhost:8501
```

## 📸 How to Use

### Step 1: Prepare Reference Photographs

Create a local folder containing individually named photographs.

Example:

```text
data/
└── reference_faces/
    ├── Person_01.jpg
    ├── Person_02.jpg
    ├── Person_03.jpg
    ├── Person_04.jpg
    └── Person_05.jpg
```

Each photograph should contain exactly one clearly visible face.

The filename, without the extension, becomes the enrolled identity.

### Step 2: Enroll Reference Faces

1. Enter the local reference folder path.
2. Click **Build Reference Database**.
3. Review the total images, enrolled identities and skipped images.
4. Save the generated embedding database.

The application stores the embeddings locally.

### Step 3: Upload a Group Photograph

1. Navigate to the group photograph upload section.
2. Upload a JPG or PNG image.
3. Click **Recognize People**.
4. Wait for face detection and matching to complete.

### Step 4: Review Recognition Results

The application displays:

- Annotated group photograph.
- Detected faces and bounding boxes.
- Recognized identity or `Unknown`.
- Similarity scores.
- Recognition results in a table.
- CSV export of recognition results.

## 🧪 Validation and Testing

The system was evaluated using a controlled subset of the Labeled Faces in the Wild (LFW) dataset.

### Known-Person Recognition

| Metric | Result |
|---|---:|
| Enrolled identities | 50 |
| Independent test photographs | 50 |
| Correct predictions | 49 |
| Incorrect predictions | 1 |
| Recognition accuracy | **98.00%** |

Each test photograph was different from its corresponding reference photograph.

### Unknown-Person Rejection

| Metric | Result |
|---|---:|
| Unknown identities | 20 |
| Unknown test photographs | 20 |
| Correctly rejected | 19 |
| False identifications | 1 |
| Unknown rejection rate | **95.00%** |

### Group Photograph Test

| Metric | Result |
|---|---:|
| Faces detected | 7 |
| Recognition results generated | 7 |
| Faces labelled Unknown | 7 |

All seven faces were detected and processed successfully in the demonstration group photograph.

### Threshold Experiment

The initial similarity threshold was set to `0.363`.

Additional threshold and confidence-margin experiments were performed to investigate the trade-off between genuine-person recognition and unknown-person rejection.

The production matcher retains the initial threshold because the small validation sample did not establish a sufficiently reliable improvement from changing it.

**Evaluation limitation:** These are preliminary results from a small controlled LFW sample. They do not represent guaranteed real-world accuracy or performance on an unseen company dataset.

## 🔐 Privacy and Security

Privacy is a core design consideration.

- Photographs are processed locally.
- Face embeddings are generated locally.
- The application does not call online face recognition APIs.
- Reference embeddings are stored in a local compressed database.
- Reference photographs and datasets are not included in the GitHub repository.

Users should protect local photographs and embedding databases because face embeddings are sensitive biometric information.

## 📚 Models and Attribution

This project uses pretrained models from OpenCV Zoo:

**YuNet — Face Detection**

- Model: `face_detection_yunet_2023mar.onnx`
- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- License: MIT

**SFace — Face Recognition**

- Model: `face_recognition_sface_2021dec.onnx`
- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- License: Apache License 2.0

Refer to [MODEL_LICENSES.md](MODEL_LICENSES.md) for model attribution and licensing information.

## ⚠️ Limitations and Future Improvements

### Current Limitations

- Recognition performance depends on image quality, lighting, face orientation and resolution.
- Each reference photograph must contain exactly one detectable face.
- The current reference database uses one reference embedding per identity.
- Similar-looking faces and difficult image conditions can produce false matches or false rejections.
- The validation sample is limited and does not establish generalization to arbitrary real-world photographs.

### Future Improvements

- Support multiple reference photographs per identity.
- Improve confidence calibration using larger independent datasets.
- Add more comprehensive unknown-person testing.
- Improve handling of low-resolution and partially occluded faces.
- Add batch image processing.
- Add better reference database management.
- Expand performance and usability testing.

## 👨‍💻 Author

**Deepak Kumar**

IIT Madras — BS in Data Science and Applications

GitHub: https://github.com/22F3000107

Project Repository:

https://github.com/22F3000107/industry-ai-offline-face-recognition

---

*Built as part of the Industry.AI Machine Learning Internship Assignment.*