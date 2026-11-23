
from pathlib import Path
import cv2
import pandas as pd

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_matcher import FaceMatcher
from src.database import ReferenceDatabase


BASE_DIR = Path("data/lfw_validation")
REFERENCE_DIR = BASE_DIR / "reference"
TEST_DIR = BASE_DIR / "test"

MODEL_DIR = Path("models")

# Initialize the existing recognition components
detector = FaceDetector(
    str(MODEL_DIR / "face_detection_yunet_2023mar.onnx")
)

encoder = FaceEncoder(
    str(MODEL_DIR / "face_recognition_sface_2021dec.onnx")
)

# Build reference database from the five LFW reference images
# Build reference database manually, selecting the largest detected face.
database = ReferenceDatabase(detector, encoder)
database.identities = {}

for image_path in sorted(REFERENCE_DIR.glob("*.jpg")):
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    faces = detector.detect(image)

    if not faces:
        print("No face detected:", image_path.name)
        continue

    # Select the largest detected face.
    face = max(
        faces,
        key=lambda item: item["bbox"][2] * item["bbox"][3]
    )

    embedding = encoder.encode(image, face["face_data"])

    if embedding is not None:
        database.identities[image_path.stem] = embedding
        print("Loaded:", image_path.stem)

print("\nReference identities:", len(database.identities))

print("\nLFW Recognition Validation")
print("=" * 60)
print("Reference identities:", len(database.identities))

# Initialize matcher
matcher = FaceMatcher(database.identities, threshold=0.363)

results = []

for person_dir in sorted(TEST_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    expected_name = person_dir.name

    for image_path in sorted(person_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path)
            continue

        faces = detector.detect(image)

        if not faces:
            print("No face detected:", image_path)
            results.append({
                "image": image_path.name,
                "expected": expected_name,
                "predicted": "No face detected",
                "similarity": None,
                "correct": False,
            })
            continue

        # LFW images are centered; use the largest detected face
        face = max(
            faces,
            key=lambda item: item["bbox"][2] * item["bbox"][3]
        )

        embedding = encoder.encode(image, face["face_data"])

        if embedding is None:
            print("Encoding failed:", image_path)
            continue

        prediction = matcher.match(embedding)

        predicted_name = prediction["identity"]
        similarity = prediction["similarity"]
        correct = predicted_name == expected_name

        results.append({
            "image": image_path.name,
            "expected": expected_name,
            "predicted": predicted_name,
            "similarity": similarity,
            "correct": correct,
        })

        print(f"\nImage: {expected_name}/{image_path.name}")
        print("Expected:", expected_name)
        print("Predicted:", predicted_name)
        print(f"Similarity: {similarity:.4f}")
        print("Result:", "PASS" if correct else "FAIL")

# Summary
df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

if not df.empty:
    total = len(df)
    correct = int(df["correct"].sum())

    print("Total test images:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)
    print(f"Accuracy: {correct / total * 100:.2f}%")

    print("\nResults by identity:")
    print(
        df.groupby("expected")["correct"]
        .agg(["sum", "count"])
        .to_string()
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "lfw_validation_results.csv"
    df.to_csv(output_file, index=False)

    print("\nDetailed results saved to:", output_file)
else:
    print("No test results generated.")
