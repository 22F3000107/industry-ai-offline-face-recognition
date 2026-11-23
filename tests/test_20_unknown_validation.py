from pathlib import Path
import cv2
import pandas as pd

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_matcher import FaceMatcher

REFERENCE_DIR = Path("data/test_50_references")
UNKNOWN_DIR = Path("data/lfw_validation/unknown_50")
THRESHOLD = 0.363

detector = FaceDetector(
    "models/face_detection_yunet_2023mar.onnx"
)

encoder = FaceEncoder(
    "models/face_recognition_sface_2021dec.onnx"
)

# Build the 50-person reference database.
identities = {}

print("Loading 50 reference identities...")
for path in sorted(REFERENCE_DIR.glob("*.jpg")):
    image = cv2.imread(str(path))

    if image is None:
        continue

    faces = detector.detect(image)
    if not faces:
        print("No reference face detected:", path.name)
        continue

    face = max(
        faces,
        key=lambda item: item["bbox"][2] * item["bbox"][3]
    )

    embedding = encoder.encode(image, face["face_data"])

    if embedding is not None:
        identities[path.stem] = embedding

print("\nReference identities loaded:", len(identities))

if len(identities) != 50:
    raise RuntimeError(f"Expected 50 identities, got {len(identities)}")

matcher = FaceMatcher(identities, threshold=THRESHOLD)
results = []

print("\nTesting 20 previously unenrolled identities...")
print("=" * 75)

for person_dir in sorted(UNKNOWN_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    for image_path in sorted(person_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        faces = detector.detect(image)

        if not faces:
            results.append({
                "actual_identity": person_dir.name,
                "predicted": "No face detected",
                "similarity": None,
                "status": "NO FACE DETECTED",
            })
            print(person_dir.name, "-> NO FACE DETECTED")
            continue

        face = max(
            faces,
            key=lambda item: item["bbox"][2] * item["bbox"][3]
        )

        embedding = encoder.encode(image, face["face_data"])

        if embedding is None:
            results.append({
                "actual_identity": person_dir.name,
                "predicted": "Encoding failed",
                "similarity": None,
                "status": "ENCODING FAILED",
            })
            continue

        prediction = matcher.match(embedding)
        predicted = prediction["identity"]
        similarity = prediction["similarity"]

        if predicted == "Unknown":
            status = "PASS"
        else:
            status = "FALSE IDENTIFICATION"

        results.append({
            "actual_identity": person_dir.name,
            "predicted": predicted,
            "similarity": similarity,
            "status": status,
        })

        print(
            f"{person_dir.name:35} -> "
            f"{predicted:35} "
            f"Score: {similarity:.4f} {status}"
        )

df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("20-UNKNOWN-PERSON VALIDATION SUMMARY")
print("=" * 75)

if not df.empty:
    total = len(df)
    rejected = int((df["status"] == "PASS").sum())
    false_ids = int((df["status"] == "FALSE IDENTIFICATION").sum())
    no_face = int((df["status"] == "NO FACE DETECTED").sum())
    encoding_failed = int((df["status"] == "ENCODING FAILED").sum())
    detected = total - no_face - encoding_failed

    print("Unknown test images:", total)
    print("Faces successfully processed:", detected)
    print("Correctly rejected:", rejected)
    print("False identifications:", false_ids)
    print("No face detected:", no_face)
    print("Encoding failures:", encoding_failed)

    if detected:
        print(
            f"Unknown rejection rate among processed faces: "
            f"{rejected / detected * 100:.2f}%"
        )

    Path("outputs").mkdir(exist_ok=True)
    output_file = Path("outputs/20_unknown_validation_results.csv")
    df.to_csv(output_file, index=False)

    print("\nDetailed results saved to:", output_file)
else:
    print("No test results generated.")
