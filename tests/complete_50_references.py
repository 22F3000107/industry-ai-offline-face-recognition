
from pathlib import Path
import shutil

from src.face_detector import FaceDetector

SOURCE = Path("data/lfw/lfw-deepfunneled/lfw-deepfunneled")
DESTINATION = Path("data/test_50_references")

detector = FaceDetector()

# Keep all existing filenames so we don't duplicate identities.
existing_names = {
    path.stem for path in DESTINATION.glob("*.jpg")
}

replacements = []

print("Searching for six valid replacement identities...")

for person_dir in sorted(SOURCE.iterdir()):
    if not person_dir.is_dir():
        continue

    person = person_dir.name

    if person in existing_names:
        continue

    images = sorted(person_dir.glob("*.jpg"))

    # Keep identities with at least two different photographs.
    if len(images) < 2:
        continue

    image = __import__("cv2").imread(str(images[0]))

    if image is None:
        continue

    faces = detector.detect(image)

    # Only accept reference photos with exactly one detected face.
    if len(faces) == 1:
        replacements.append((person, images[0]))

    if len(replacements) == 6:
        break

if len(replacements) < 6:
    raise RuntimeError(
        f"Only found {len(replacements)} valid replacements."
    )

for person, image_path in replacements:
    destination = DESTINATION / f"{person}.jpg"
    shutil.copy2(image_path, destination)
    print("Added:", person)

print("\nSix replacement references added.")
print(
    "Total reference images:",
    len(list(DESTINATION.glob("*.jpg")))
)
