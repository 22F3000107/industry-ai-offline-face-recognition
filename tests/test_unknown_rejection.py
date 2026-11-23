
from pathlib import Path
import cv2
import pandas as pd

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_matcher import FaceMatcher

REFERENCE_DIR = Path("data/lfw_validation/reference")
UNKNOWN_DIR = Path("data/lfw_validation/unknown_test")

detector = FaceDetector(
    "models/face_detection_yunet_2023mar.onnx"
)

encoder = FaceEncoder(
    "models/face_recognition_sface_2021dec.onnx"
)

# Build the same five-person reference database
identities = {}

for path in sorted(REFERENCE_DIR.glob("*.jpg")):
    image = cv2.imread(str(path))

    if image is None:
        continue

    faces = detector.detect(image)

    if not faces:
        print("No reference face detected:", path.name)
        continue

    # Select the largest face
    face = max(
        faces,
        key=lambda item: item["bbox"][2] * item["bbox"][3]
    )

    embedding = encoder.encode(image, face["face_data"])

    if embedding is not None:
        identities[path.stem] = embedding

print("\nUnknown Person Rejection Test")
print("=" * 60)
print("Reference identities:", len(identities))

matcher = FaceMatcher(identities, threshold=0.363)

results = []

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
                "correct_rejection": False,
            })
            continue

        face = max(
            faces,
            key=lambda item: item["bbox"][2] * item["bbox"][3]
        )

        embedding = encoder.encode(image, face["face_data"])

        if embedding is None:
            continue

        prediction = matcher.match(embedding)

        predicted = prediction["identity"]
        similarity = prediction["similarity"]

        rejected = predicted == "Unknown"

        results.append({
            "actual_identity": person_dir.name,
            "predicted": predicted,
            "similarity": similarity,
            "correct_rejection": rejected,
        })

        print(f"\nImage: {person_dir.name}/{image_path.name}")
        print("Expected: Unknown")
        print("Predicted:", predicted)
        print(f"Similarity: {similarity:.4f}")
        print("Result:", "PASS" if rejected else "FALSE IDENTIFICATION")

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("UNKNOWN REJECTION SUMMARY")
print("=" * 60)

if not df.empty:
    total = len(df)
    rejected = int(df["correct_rejection"].sum())
    false_identifications = total - rejected

    print("Total images:", total)
    print("Correctly rejected:", rejected)
    print("False identifications:", false_identifications)
    print(f"Unknown rejection rate: {rejected / total * 100:.2f}%")

    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/unknown_rejection_results.csv", index=False)
    print("\nSaved: outputs/unknown_rejection_results.csv")
