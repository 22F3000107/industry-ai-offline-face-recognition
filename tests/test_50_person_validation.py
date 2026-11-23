from pathlib import Path
import cv2
import pandas as pd

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_matcher import FaceMatcher

REFERENCE_DIR = Path("data/test_50_references")
TEST_DIR = Path("data/lfw_validation/test_50")
MODEL_DIR = Path("models")

detector = FaceDetector(
    str(MODEL_DIR / "face_detection_yunet_2023mar.onnx")
)

encoder = FaceEncoder(
    str(MODEL_DIR / "face_recognition_sface_2021dec.onnx")
)

# Build the reference database using the 50 enrolled reference photographs.
identities = {}

print("Building 50-person reference database...")
print("=" * 60)

for image_path in sorted(REFERENCE_DIR.glob("*.jpg")):
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    faces = detector.detect(image)

    if not faces:
        print("No face detected:", image_path.name)
        continue

    face = max(
        faces,
        key=lambda item: item["bbox"][2] * item["bbox"][3]
    )

    embedding = encoder.encode(image, face["face_data"])

    if embedding is not None:
        identities[image_path.stem] = embedding
        print("Loaded:", image_path.stem)
    else:
        print("Encoding failed:", image_path.name)

print("\nReference identities loaded:", len(identities))

if len(identities) != 50:
    raise RuntimeError(
        f"Expected 50 reference identities, but loaded {len(identities)}"
    )

matcher = FaceMatcher(identities, threshold=0.363)
results = []

print("\nStarting independent 50-person validation...")
print("=" * 60)

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
            results.append({
                "image": str(image_path),
                "expected": expected_name,
                "predicted": "No face detected",
                "similarity": None,
                "correct": False,
            })
            print("NO FACE:", expected_name)
            continue

        face = max(
            faces,
            key=lambda item: item["bbox"][2] * item["bbox"][3]
        )

        embedding = encoder.encode(image, face["face_data"])

        if embedding is None:
            results.append({
                "image": str(image_path),
                "expected": expected_name,
                "predicted": "Encoding failed",
                "similarity": None,
                "correct": False,
            })
            print("ENCODING FAILED:", expected_name)
            continue

        prediction = matcher.match(embedding)

        predicted_name = prediction["identity"]
        similarity = prediction["similarity"]
        correct = predicted_name == expected_name

        results.append({
            "image": str(image_path),
            "expected": expected_name,
            "predicted": predicted_name,
            "similarity": similarity,
            "correct": correct,
        })

        print(
            f"{expected_name:35} -> "
            f"{predicted_name:35} "
            f"Score: {similarity:.4f} "
            f"{'PASS' if correct else 'FAIL'}"
        )

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("50-PERSON VALIDATION SUMMARY")
print("=" * 60)

if not df.empty:
    total = len(df)
    correct = int(df["correct"].sum())

    print("Reference identities:", len(identities))
    print("Test images:", total)
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

    output_file = output_dir / "50_person_validation_results.csv"
    df.to_csv(output_file, index=False)

    print("\nDetailed results saved to:", output_file)
else:
    print("No test results generated.")
