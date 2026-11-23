# Research Notes: Who Is in the Photo?

## 1. Scope and research problem

This project explores **offline face recognition in group photographs**. Given a set of locally supplied reference photographs and a new photograph containing one or more people, it detects faces, compares each detected face with the enrolled identities, and either returns the closest identity or `Unknown`.

The scope described here is the implementation and the validation artifacts present in this repository. Results are preliminary measurements on small, controlled image sets. They should not be interpreted as expected accuracy on arbitrary photographs, a demographic fairness assessment, or a production-ready biometric evaluation.

## 2. Implemented pipeline and model choices

The application uses Streamlit for its interface and OpenCV for image processing and model inference. It loads two local ONNX files: `models/face_detection_yunet_2023mar.onnx` and `models/face_recognition_sface_2021dec.onnx` ([`app.py`](app.py#L26-L31), [`requirements.txt`](requirements.txt)). The UI says photographs and embeddings are processed locally and that no online recognition APIs are used ([`app.py`](app.py#L292-L295)). The repository describes this as offline processing; this is an implementation property, not a measured accuracy result.

### 2.1 Face detection: YuNet

`FaceDetector` constructs OpenCV's `FaceDetectorYN` from the local YuNet ONNX model. Its configured defaults are score threshold `0.7`, NMS threshold `0.3`, and `top_k=5000`; before detection it sets the input size to the image dimensions ([`src/face_detector.py`](src/face_detector.py#L9-L30), [`src/face_detector.py`](src/face_detector.py#L47-L53)). The detector returns bounding boxes and the YuNet face data used by the encoder.

YuNet is the detector selected and implemented in this project. Its paper describes a compact, fast face detector intended for constrained and edge settings, which fits the project's local-inference goal; the OpenCV `FaceDetectorYN` API and Zoo model make it directly usable in this OpenCV/ONNX pipeline ([Wu, Peng, and Yu, “YuNet: A Tiny Millisecond-level Face Detector”](https://link.springer.com/article/10.1007/s11633-023-1423-y); [OpenCV Zoo YuNet model page](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)). Those sources explain the implementation fit; the repository contains **no head-to-head detector experiment** establishing that YuNet outperforms another detector for this dataset.

### 2.2 Face recognition: SFace

`FaceEncoder` constructs OpenCV's `FaceRecognizerSF` with the local SFace ONNX model. It aligns the detected face using YuNet's 15-value detection record, extracts a feature, converts it to a flattened `float32` NumPy vector, and L2-normalizes it ([`src/face_encoder.py`](src/face_encoder.py#L7-L24), [`src/face_encoder.py`](src/face_encoder.py#L26-L63)).

SFace is the recognition model selected by the implementation. It fits the project's local pipeline because OpenCV provides the `FaceRecognizerSF` API for aligning faces and extracting features from the local model; the SFace paper describes its sigmoid-constrained hypersphere loss for face recognition ([Zhong et al., “SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition”](https://ieeexplore.ieee.org/document/9318547)). OpenCV Zoo documents the SFace model artifact ([OpenCV Zoo SFace model page](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)); OpenCV's tutorial documents the YuNet detection, SFace alignment/feature extraction, and cosine-comparison workflow ([OpenCV face detection and recognition tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)). The project does **not** include a comparative experiment against another recognizer; therefore these papers and upstream docs are method context, not evidence of a locally measured advantage.

### 2.3 OpenCV, ONNX, and local inference

The model constructors receive local filesystem paths, and inference is performed through OpenCV's APIs in the Python process. ONNX is the serialized model format, designed to support exchange of machine-learning models between frameworks and runtimes ([ONNX overview](https://onnx.ai/onnx/repo-docs/Overview.html)). The project uses those local artifacts with OpenCV; it does not call a hosted inference service or cloud recognition API. This statement concerns inference and processing in this application, not every possible network operation on the machine.

## 3. Reference enrollment and embeddings

The app lets the user select a local reference folder and build a reference database ([`app.py`](app.py#L338-L368)). `ReferenceDatabase.build()` reads JPG, JPEG, and PNG files; the filename stem becomes the identity label; and an image is accepted only when exactly one face is detected. The accepted face is encoded and stored in the in-memory `identities` mapping ([`src/database.py`](src/database.py#L18-L78)).

`ReferenceDatabase.save()` can write the identity names and stacked embeddings to a compressed local NumPy `.npz` file ([`src/database.py`](src/database.py#L80-L100)). The class also implements `load()` ([`src/database.py`](src/database.py#L102-L118)). The Streamlit app has an optional save action; its recognition flow builds the reference database in the current session. The report does not imply that a saved archive is automatically loaded by the app on every run.

## 4. Matching and Unknown-person rejection

The encoder normalizes the query embedding. `FaceMatcher` also validates and normalizes the query and each reference embedding, computes the dot product for each identity, selects the highest score, and accepts that identity only when its score is at least the configured threshold. Otherwise it returns `Unknown` ([`src/face_matcher.py`](src/face_matcher.py#L23-L75)). Because both vectors are normalized, their dot product is cosine similarity.

The production/default threshold is **`0.363`** in the app and matcher ([`app.py`](app.py#L31), [`src/face_matcher.py`](src/face_matcher.py#L8-L20)). The Streamlit control allows a user to adjust the threshold interactively, and the UI advises validating it on representative data ([`app.py`](app.py#L305-L317)). This report does not change that value. OpenCV Zoo's SFace example also uses cosine distance/similarity conventions and documents a `0.363` reference threshold ([OpenCV Zoo SFace example](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py)); the repository's own matcher correctly applies the threshold to similarity as `best_score >= threshold`.

## 5. Validation method and measured results

The repository has standalone validation scripts and saved CSV outputs rather than a single automated benchmark suite. The 50-person script loads 50 reference images and evaluates one test image for each of the same 50 identities. It selects the largest detected face in each image and creates a `FaceMatcher` at `0.363` ([`tests/test_50_person_validation.py`](tests/test_50_person_validation.py)). The unknown-person script uses the same 50 references and 20 images from identities outside that enrolled set, also selecting the largest detected face and using `0.363` ([`tests/test_20_unknown_validation.py`](tests/test_20_unknown_validation.py)). These are face-image classification/rejection checks; they do not constitute a full multi-face group-photo benchmark.

### 5.1 Known-person validation

The recorded result is **49/50 correct (98%)** in [`outputs/50_person_validation_results.csv`](outputs/50_person_validation_results.csv). The single error is:

| Expected identity | System result | Best similarity | Interpretation |
|---|---|---:|---|
| `Aldo_Paredes` | `Unknown` | `0.220127` | False rejection: the genuine identity did not reach `0.363`. The score table shows `Abdoulaye_Wade` as the highest-scoring candidate at `0.220127`, with `Akhmed_Zakayev` second at `0.206285`; both are below threshold. |

Thus, this one error is a rejected known face, not an accepted wrong identity.

### 5.2 Unknown-person validation

The recorded result is **19/20 correctly rejected (95%)** in [`outputs/20_unknown_validation_results.csv`](outputs/20_unknown_validation_results.csv). Its single error is:

| Actual identity | System result | Best similarity | Interpretation |
|---|---|---:|---|
| `Alvaro_Silva_Calderon` | `Albert_Costa` | `0.436156` | False acceptance: the unknown face exceeded `0.363` and was assigned to an enrolled identity. |

The percentages are the observed proportions in these CSVs, not confidence guarantees. Both samples are small and preliminary.

### 5.3 Other saved validation outputs

Separate scripts and CSVs report 10/10 correct on a five-identity custom subset ([`tests/test_lfw_validation.py`](tests/test_lfw_validation.py), [`outputs/lfw_validation_results.csv`](outputs/lfw_validation_results.csv)) and 10/10 rejections in a separate five-reference unknown test ([`tests/test_unknown_rejection.py`](tests/test_unknown_rejection.py), [`outputs/unknown_rejection_results.csv`](outputs/unknown_rejection_results.csv)). These are distinct small checks and are not pooled into the 50/20 results above.

The project uses images described as LFW, but the scripts select custom reference, known-test, and unknown-test folders and a small set of identities. This is **not the official LFW verification benchmark/protocol** and should be called a custom LFW subset validation. The original LFW dataset paper describes the dataset and its unconstrained face-recognition evaluation context ([Huang et al., “Labeled Faces in the Wild”](https://cs.umass.edu/~elm/papers/lfw.pdf)).

## 6. Threshold and confidence-margin analysis

### 6.1 Confidence-margin experiment

`tests/test_confidence_margin.py` records the best and second-best candidate scores for the same 50 known and 20 unknown samples, then evaluates margin limits `0.00`, `0.01`, `0.02`, `0.03`, `0.04`, `0.05`, `0.06`, `0.08`, and `0.10` while holding the similarity threshold at `0.363`. A sample is accepted only if both its best score reaches the threshold and `best_score - second_score` reaches the tested margin ([script](tests/test_confidence_margin.py), [recorded scores](outputs/confidence_margin_scores.csv)).

Margins through `0.04` leave the headline counts unchanged: 49/50 known identities correct and 19/20 unknown people rejected. At `0.05` and higher, known correct results fall to 48/50 while unknown rejection stays 19/20. `Adam_Scott` is the known sample affected at `0.05`: its best score is `0.372839` (above the similarity threshold), but its margin is only `0.048469`; applying the margin would turn this otherwise correct match into a rejection. The unknown false match, `Alvaro_Silva_Calderon`, has a margin of about `0.155620`, so none of the tested margins through `0.10` rejects it.

For these reasons, the confidence-margin rule was **not adopted**: the tested settings did not improve unknown rejection, and stricter tested margins reduced known-person correct results. The production matcher continues to use its existing similarity-threshold decision, without a confidence-margin requirement.

### 6.2 Post-hoc threshold re-scoring (not a new experiment)

The 70 rows in `outputs/confidence_margin_scores.csv` contain each sample's best score and candidate identity. Reapplying different cutoff values to those already-recorded scores gives the following post-hoc counts:

| Similarity cutoff | Known correctly identified | Known rejected as Unknown | Unknown correctly rejected | Unknown falsely accepted |
|---:|---:|---:|---:|---:|
| 0.240 | 49/50 | 1/50 | 2/20 | 18/20 |
| 0.300 | 49/50 | 1/50 | 16/20 | 4/20 |
| **0.363 (current)** | **49/50** | **1/50** | **19/20** | **1/20** |
| 0.440 | 48/50 | 2/50 | 20/20 | 0/20 |

At `0.440`, the recorded unknown false acceptance falls below cutoff, but two known samples are rejected. At `0.363`, one unknown remains falsely accepted and one known is rejected. This is a re-scoring of stored outputs, not a fresh model inference run, a preregistered threshold experiment, or a setting change. The app and matcher remain at their existing default of `0.363`.

## 7. Limitations and dataset considerations

- The headline validation contains only 50 known test photographs and 20 unknown test photographs, one per identity in the listed split. Rates on samples this small are preliminary and sensitive to individual images.
- The known test set shares identities with the enrolled reference set by design; it measures recognizing another photograph of those enrolled people. Unknown identities are distinct from the enrolled identities. This is not an identity-disjoint generalization test for known-person recognition.
- The validation scripts use the largest detected face per image. They do not establish accuracy across every face in varied group photographs, face sizes, occlusions, image qualities, camera conditions, or demographic groups.
- The custom LFW subset and its folder selection do not follow the official LFW verification protocol. Reported figures must not be presented as official LFW benchmark results.
- The saved result CSVs are useful for reproducing the reported counts and examining errors, but they do not alone describe all dataset curation, provenance, or potential overlap risks. The repository's README calls the data controlled and preliminary; broader validation with documented, consented, representative data remains necessary.
- Similarity scores and a fixed cutoff can yield both false rejection and false acceptance, as the recorded errors demonstrate. A threshold chosen on small validation data may not transfer to a different population or image distribution.
- Face embeddings are sensitive biometric data. Local storage avoids sending them to an online recognition API in this implementation, but the files still require appropriate local access controls and retention practices.

## 8. Model licensing and provenance

The repository's [`MODEL_LICENSES.md`](MODEL_LICENSES.md) identifies both pretrained ONNX files as sourced from OpenCV Zoo and records YuNet under the MIT License and SFace under Apache License 2.0. Refer to the upstream model directories for the applicable files and full terms: [YuNet model directory and license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace model directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), including its [Apache-2.0 license file](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE). This report records the repository's attribution; it is not legal advice or a substitute for reviewing the complete upstream terms and the provenance/terms of any datasets used.

## References

1. Wu, W., Peng, H., and Yu, S. “YuNet: A Tiny Millisecond-level Face Detector.” *Machine Intelligence Research* (2023). [Publisher page](https://link.springer.com/article/10.1007/s11633-023-1423-y).
2. Zhong, Y. et al. “SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition.” *IEEE Transactions on Image Processing* (2021). [IEEE Xplore](https://ieeexplore.ieee.org/document/9318547).
3. OpenCV. “DNN-based Face Detection and Recognition.” [OpenCV documentation](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
4. ONNX. “ONNX Overview.” [ONNX documentation](https://onnx.ai/onnx/repo-docs/Overview.html).
5. Huang, G. B. et al. “Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.” [LFW paper](https://cs.umass.edu/~elm/papers/lfw.pdf).
6. OpenCV Zoo. [YuNet model notes](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [SFace model notes](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), and [SFace license](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE).
