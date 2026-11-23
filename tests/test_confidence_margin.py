from pathlib import Path
import cv2
import pandas as pd

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_matcher import FaceMatcher

REFERENCE_DIR = Path("data/test_50_references")
KNOWN_DIR = Path("data/lfw_validation/test_50")
UNKNOWN_DIR = Path("data/lfw_validation/unknown_50")

detector = FaceDetector(
    "models/face_detection_yunet_2023mar.onnx"
)

encoder = FaceEncoder(
    "models/face_recognition_sface_2021dec.onnx"
)

# Load the 50 reference identities.
identities = {}

for path in sorted(REFERENCE_DIR.glob("*.jpg")):
    image = cv2.imread(str(path))
    if image is None:
        continue

    faces = detector.detect(image)
    if not faces:
        continue

    face = max(
        faces,
        key=lambda item: item["bbox"][2] * item["bbox"][3]
    )

    embedding = encoder.encode(image, face["face_data"])

    if embedding is not None:
        identities[path.stem] = embedding

print("Reference identities:", len(identities))

if len(identities) != 50:
    raise RuntimeError("Expected exactly 50 reference identities.")

matcher = FaceMatcher(identities, threshold=0.363)


def evaluate_image(image_path, expected, is_unknown):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    faces = detector.detect(image)
    if not faces:
        return None

    face = max(
        faces,
        key=lambda item: item["bbox"][2] * item["bbox"][3]
    )

    embedding = encoder.encode(image, face["face_data"])
    if embedding is None:
        return None

    prediction = matcher.match(embedding)

    ranked = sorted(
        prediction["all_scores"].items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_name, best_score = ranked[0]
    second_name, second_score = ranked[1]

    return {
        "expected": expected,
        "is_unknown": is_unknown,
        "best_name": best_name,
        "best_score": best_score,
        "second_name": second_name,
        "second_score": second_score,
        "margin": best_score - second_score,
    }


results = []

# Evaluate known people.
for person_dir in sorted(KNOWN_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    for path in sorted(person_dir.glob("*.jpg")):
        result = evaluate_image(path, person_dir.name, False)
        if result:
            results.append(result)

# Evaluate unknown people.
for person_dir in sorted(UNKNOWN_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    for path in sorted(person_dir.glob("*.jpg")):
        result = evaluate_image(path, person_dir.name, True)
        if result:
            results.append(result)

df = pd.DataFrame(results)

print("\nProcessed images:", len(df))
print("Known images:", int((~df["is_unknown"]).sum()))
print("Unknown images:", int(df["is_unknown"].sum()))

print("\nLowest known-person margins:")
print(
    df[~df["is_unknown"]]
    .sort_values("margin")
    [["expected", "best_name", "best_score", "second_name",
      "second_score", "margin"]]
    .head(10)
    .to_string(index=False)
)

print("\nHighest unknown-person margins:")
print(
    df[df["is_unknown"]]
    .sort_values("margin", ascending=False)
    [["expected", "best_name", "best_score", "second_name",
      "second_score", "margin"]]
    .head(10)
    .to_string(index=False)
)

# Compare different confidence margins.
print("\nCONFIDENCE MARGIN COMPARISON")
print("=" * 85)
print(
    f"{'Margin':<12}{'Known Correct':<18}"
    f"{'Known Accuracy':<19}{'Unknown Rejected':<20}"
    f"{'Unknown Rate':<15}"
)

for margin_limit in [0.00, 0.01, 0.02, 0.03, 0.04,
                     0.05, 0.06, 0.08, 0.10]:

    accepted = (
        (df["best_score"] >= 0.363) &
        (df["margin"] >= margin_limit)
    )

    known = ~df["is_unknown"]
    unknown = df["is_unknown"]

    known_correct = (
        accepted & (df["best_name"] == df["expected"]) & known
    ).sum()

    unknown_rejected = (~accepted & unknown).sum()

    known_total = known.sum()
    unknown_total = unknown.sum()

    print(
        f"{margin_limit:<12.2f}"
        f"{known_correct}/{known_total:<16}"
        f"{known_correct / known_total * 100:<19.2f}"
        f"{unknown_rejected}/{unknown_total:<18}"
        f"{unknown_rejected / unknown_total * 100:.2f}%"
    )

Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/confidence_margin_scores.csv", index=False)

print("\nDetailed scores saved to outputs/confidence_margin_scores.csv")
