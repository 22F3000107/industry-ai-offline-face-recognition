
from pathlib import Path

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.database import ReferenceDatabase

folder = Path("data/lfw_validation/reference")

detector = FaceDetector(
    "models/face_detection_yunet_2023mar.onnx"
)

encoder = FaceEncoder(
    "models/face_recognition_sface_2021dec.onnx"
)

database = ReferenceDatabase(detector, encoder)
database.build(str(folder))

print("\nREFERENCE DIAGNOSTIC")
print("=" * 50)

print("\nSuccessfully loaded identities:")
for name in database.identities:
    print("-", name)

print("\nIndividual image checks:")

import cv2

for path in sorted(folder.glob("*.jpg")):
    image = cv2.imread(str(path))

    faces = detector.detect(image) if image is not None else []

    print(f"\n{path.name}")
    print("Image readable:", image is not None)
    print("Faces detected:", len(faces))

    if faces:
        embedding = encoder.encode(image, faces[0]["face_data"])
        print("Embedding generated:", embedding is not None)
