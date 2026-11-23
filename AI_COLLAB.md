# AI Collaboration Record: Who Is in the Photo?

This record summarizes AI-assisted analysis and documentation visible in the available project conversation. It distinguishes user instructions, AI contributions, and implementation evidence. A source file or validation result establishes that an artifact exists; by itself it does not establish who originally wrote the code, curated a dataset, or ran a script.

Quoted excerpts below are from user messages available in the project conversation. They are not reconstructed dialogue. Where no exact relevant excerpt is available, that is stated. The [shared ChatGPT conversation](https://chatgpt.com/share/6b043c52-003c-83ee-8144-52b547c43714) is a reference to the project discussion, not a complete transcript export; the complete ChatGPT data export has been requested and is pending.

## 1. Choosing an entirely offline architecture

**Problem or challenge.** The design needed to describe processing accurately and avoid implying that photographs or embeddings are sent to cloud services.

**AI contribution.** AI inspected the app's model paths and privacy copy and documented the local inference boundary. It did not originate the offline requirement.

**Human decision or verification.** You specified that local processing and no cloud recognition APIs were essential. You also required architecture documentation to reflect implemented behavior.

**Actual implementation.** `app.py` loads model files from local `models/` paths and says photographs and embeddings are processed locally, with no online recognition APIs ([`app.py`](app.py#L26-L31), [`app.py`](app.py#L292-L295)). The implementation uses local image processing and OpenCV inference; this is not a claim that the computer never uses a network for any purpose.

**Conversation excerpt — user instruction:** “Show that inference and image processing run locally without cloud APIs.”

## 2. Documenting YuNet and SFace

**Problem or challenge.** The architecture needed to explain which models perform detection and embedding generation without inventing model comparisons.

**AI contribution.** AI traced the model calls through the detector and encoder and described their roles. The available conversation shows you specifying these model names; it does not show AI independently selecting them after a model comparison.

**Human decision or verification.** You required YuNet detection and SFace embedding generation to appear in the architecture. No source-backed comparison of alternative detectors or recognizers was found.

**Actual implementation.** [`src/face_detector.py`](src/face_detector.py) creates OpenCV's YuNet `FaceDetectorYN` from a local ONNX path. [`src/face_encoder.py`](src/face_encoder.py) creates OpenCV's SFace `FaceRecognizerSF`, aligns a YuNet-detected face, and extracts a normalized NumPy feature. The upstream model/license attribution is recorded in [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

**Conversation excerpt — user instruction:** “Include YuNet face detection, SFace embedding generation…”

## 3. Describing reference enrollment

**Problem or challenge.** Enrollment behavior had to match the code, including the identity naming rule and one-face requirement.

**AI contribution.** AI read `ReferenceDatabase.build()` and the Streamlit flow and documented their actual data path. It did not claim to have implemented the enrollment code.

**Human decision or verification.** You requested a separate reference-enrollment workflow in the architecture deliverables and approved the proposed documentation plan before those deliverables were generated.

**Actual implementation.** The app accepts a local folder path. [`src/database.py`](src/database.py#L18-L78) reads JPG/JPEG/PNG images, derives each identity from the filename stem, requires exactly one detected face, and stores its SFace embedding in an in-memory mapping. The app places that mapping in session state and offers an optional save action to a compressed NumPy `.npz` file ([`app.py`](app.py#L338-L368), [`app.py`](app.py#L390-L400)). The class has a `load()` method, but the app's recognition flow does not automatically call it ([`src/database.py`](src/database.py#L102-L118)).

**Conversation excerpt — user instruction:** “Show the reference enrollment workflow…”

## 4. Explaining matching and Unknown rejection

**Problem or challenge.** The system needed to distinguish a best candidate from an accepted identity and explain how unfamiliar people can be rejected.

**AI contribution.** AI traced the score calculation and threshold branch and explained that a normalized-vector dot product implements cosine similarity. It also used CSV evidence to distinguish a false rejection from a false identification.

**Human decision or verification.** You asked that cosine similarity and threshold-based `Unknown` rejection be represented and later required that the production threshold remain unchanged.

**Actual implementation.** [`src/face_matcher.py`](src/face_matcher.py#L23-L75) normalizes query and reference embeddings, selects the highest dot-product score, and returns the corresponding identity when the score is at least the supplied threshold; otherwise it returns `Unknown`. The app's initial threshold is `0.363` ([`app.py`](app.py#L31)).

**Conversation excerpt — user instruction:** “…cosine similarity matching, threshold-based Unknown rejection…”

## 5. Documenting the Streamlit interface

**Problem or challenge.** The architecture needed to show the real user entry point and how users reach enrollment, recognition, and results.

**AI contribution.** AI mapped Streamlit's controls and event paths to the processing classes while limiting the diagram to existing UI behavior.

**Human decision or verification.** You explicitly required Streamlit to be shown as the user interface and prohibited invented functionality.

**Actual implementation.** [`app.py`](app.py) configures a Streamlit page, exposes a threshold slider, provides a reference-folder control and build action, accepts an uploaded group photo, and displays recognition results. It also offers a CSV download for results ([`app.py`](app.py#L20-L24), [`app.py`](app.py#L302-L324), [`app.py`](app.py#L342-L368), [`app.py`](app.py#L411-L437), [`app.py`](app.py#L513-L515)).

**Conversation excerpts — user instructions:** “Show Streamlit as the user interface.” “Do not invent components or features that are not implemented.”

## 6. Reviewing the 50-person validation dataset and result

**Problem or challenge.** A research summary needed to report the known-person result from actual scripts and saved outputs and avoid overstating what the sample measures.

**AI contribution.** AI inspected the validation script and CSV and verified the recorded count: 49 correct results out of 50 (98%). It noted that the script uses the largest detected face and tests known identities also present in the reference set. AI did not claim to have curated or originally run this dataset.

**Human decision or verification.** You required inspection of the actual scripts and CSVs and supplied the expected result for checking. The available conversation does not establish who selected each dataset image or who originally executed the validation script.

**Actual implementation and evidence.** [`tests/complete_50_references.py`](tests/complete_50_references.py) searches LFW identity folders for replacement references with at least two photographs and exactly one detected face in the chosen reference. [`tests/test_50_person_validation.py`](tests/test_50_person_validation.py) uses 50 reference identities, evaluates the corresponding known-person test folders at `0.363`, and writes [`outputs/50_person_validation_results.csv`](outputs/50_person_validation_results.csv). That CSV has 49 correct entries and one incorrect entry. This is a small custom LFW subset check, not the official LFW benchmark.

**Conversation excerpt — user-provided expected result:** “Known-person validation: 49/50 correct (98%).”

## 7. Investigating the known-person false rejection

**Problem or challenge.** The aggregate known-person score concealed what happened to the one failed sample.

**AI contribution.** AI inspected the relevant CSV row and described the failure as a false rejection rather than a wrong identity accepted by the system.

**Human decision or verification.** You asked for actual measured results and required the individual errors and similarity scores to be explained. The underlying value is directly present in the CSV.

**Actual implementation and evidence.** In [`outputs/50_person_validation_results.csv`](outputs/50_person_validation_results.csv), `Aldo_Paredes` is predicted as `Unknown` at best similarity `0.220127`, below the threshold `0.363`. The detailed score record in [`outputs/confidence_margin_scores.csv`](outputs/confidence_margin_scores.csv) shows `Abdoulaye_Wade` as the best candidate at `0.220127` and `Akhmed_Zakayev` second at `0.206285`. This is a rejected known face, not an accepted false identity.

**Conversation excerpt — user instruction:** “Clearly explain the individual errors and their similarity scores.”

## 8. Evaluating the unknown-person false identification

**Problem or challenge.** Correctly rejecting most unknown samples did not explain the one unknown that crossed the threshold.

**AI contribution.** AI matched the error row to the prediction and score, and described it as a false acceptance of an enrolled identity.

**Human decision or verification.** You requested the unknown-person result and error details. The CSV provides direct evidence for the case.

**Actual implementation and evidence.** [`tests/test_20_unknown_validation.py`](tests/test_20_unknown_validation.py) uses the 50-person reference set and tests 20 unenrolled identities. [`outputs/20_unknown_validation_results.csv`](outputs/20_unknown_validation_results.csv) records 19/20 correctly rejected (95%). The false identification is `Alvaro_Silva_Calderon` predicted as `Albert_Costa` at similarity `0.436156`, above the `0.363` threshold.

**Conversation excerpt — user-provided expected result:** “Unknown-person validation: 19/20 correctly rejected (95%).”

## 9. Reviewing threshold and confidence-margin alternatives

**Problem or challenge.** The project needed to explain whether alternative cutoffs or a best-versus-second-best confidence margin improved rejection without harming known-person recognition.

**AI contribution.** AI inspected the implemented margin sweep and re-scored the existing 70 recorded best-match scores at alternate similarity cutoffs. The alternate-cutoff table is explicitly post-hoc analysis of saved scores, not a new inference experiment. AI also identified the samples responsible for the margin trade-off.

**Human decision or verification.** You explicitly directed that the production threshold stay at `0.363` and that application code and settings not be changed. No conversation evidence shows a human adopting a margin rule or a new production threshold.

**Actual implementation and evidence.** [`tests/test_confidence_margin.py`](tests/test_confidence_margin.py) evaluates margins from `0.00` through `0.10` while keeping the similarity cutoff at `0.363`; the production [`src/face_matcher.py`](src/face_matcher.py) has no margin condition. Margins through `0.04` preserve 49/50 known correct and 19/20 unknown rejected. At `0.05` and above, known correct falls to 48/50 while unknown rejection remains 19/20. The affected known result is `Adam_Scott`, with best score `0.372839` and margin `0.048469`. The unknown false match has margin about `0.155620`, outside all tested margin limits. The post-hoc similarity re-scoring shows that `0.440` rejects all 20 unknown samples but also rejects two known samples; at the implemented `0.363`, one unknown is falsely accepted and one known is rejected. These results explain why the production cutoff remains at `0.363`: no reviewed alternative demonstrated an acceptable improvement on this small sample, and the user explicitly required retaining it. This is not a claim that `0.363` is universally optimal.

**Conversation excerpts — user instructions:** “Experiments with alternative thresholds and confidence margins.” “Keep the production threshold at 0.363. Do not change application code or threshold settings.”

## 10. Using Archify to document the architecture

**Problem or challenge.** The system needed a high-level architecture diagram and editable/explorable documentation that remained source-backed and readable.

**AI contribution.** After authorization, AI inspected the application and created the Archify candidate, standalone SVG, interactive HTML, and `ARCHITECTURE.md`. AI then revised the diagram layout and ran visual checks as requested. These are documentation artifacts; Archify did not create the application's runtime components.

**Human decision or verification.** You approved the proposed documentation deliverables, requested a clearer separation between enrollment and recognition, and asked for light- and dark-mode preview inspection. The conversation records human approval and visual-review instructions, not the user's execution of the validation scripts.

**Actual outcome.** [`ARCHITECTURE.md`](ARCHITECTURE.md) points to the SVG and documents the source-backed workflows. The interactive HTML and editable candidate are in [`.archify/architecture-who-is-in-the-photo-20261123-1743/`](.archify/architecture-who-is-in-the-photo-20261123-1743/). Review outputs are retained in that folder's `review-*` subdirectories.

**Conversation excerpts — user instructions:** “Use the installed Archify skill to analyze this existing repository…” “Please improve its visual layout before finalizing.”

## Human review and verification

The available project conversation records your approval of the proposed architecture deliverables, requests to correct diagram layout, and a request to inspect light- and dark-mode previews. For the research summary, you required source and CSV inspection, specified the results and error reporting needs, and explicitly kept the threshold at `0.363`. These are evidence of human direction and review of the documentation work.

The scripts and CSV files were inspected in the collaboration, and their reported counts and failure cases were checked against the saved rows. The conversation does **not** establish that you personally reran those scripts or independently recalculated those values. This record therefore does not claim that you did so. Nor does it attribute the pre-existing application implementation or dataset curation to AI without evidence.

## Evidence map

- Application and processing behavior: [`app.py`](app.py), [`src/face_detector.py`](src/face_detector.py), [`src/face_encoder.py`](src/face_encoder.py), [`src/face_matcher.py`](src/face_matcher.py), [`src/database.py`](src/database.py).
- Validation setup: [`tests/complete_50_references.py`](tests/complete_50_references.py), [`tests/test_50_person_validation.py`](tests/test_50_person_validation.py), [`tests/test_20_unknown_validation.py`](tests/test_20_unknown_validation.py), [`tests/test_confidence_margin.py`](tests/test_confidence_margin.py).
- Measured outputs: [`outputs/50_person_validation_results.csv`](outputs/50_person_validation_results.csv), [`outputs/20_unknown_validation_results.csv`](outputs/20_unknown_validation_results.csv), [`outputs/confidence_margin_scores.csv`](outputs/confidence_margin_scores.csv).
- Architecture and model attribution: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`MODEL_LICENSES.md`](MODEL_LICENSES.md), and the [Archify project artifacts](.archify/architecture-who-is-in-the-photo-20261123-1743/).
- Conversation evidence: [shared ChatGPT project conversation](https://chatgpt.com/share/6b043c52-003c-83ee-8144-52b547c43714). This shared link is not a complete exported transcript; the requested data export is pending. No separate transcript or AI decision log is stored in the repository.
