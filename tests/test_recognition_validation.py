
import cv2

from src.database import ReferenceDatabase
from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_matcher import FaceMatcher


def main():
    database = ReferenceDatabase()
    database.load("data/reference_embeddings.npz")

    detector = FaceDetector()
    encoder = FaceEncoder()

    matcher = FaceMatcher(
        database.identities,
        threshold=0.363
    )

    folder = "data/reference_faces"

    print("\nReference Identity Validation")
    print("=============================")

    for name in database.identities:
        image_path = f"{folder}/{name}.jpg"

        image = cv2.imread(image_path)

        # Also support PNG or JPEG extensions.
        if image is None:
            from pathlib import Path

            matches = [
                p for p in Path(folder).iterdir()
                if p.stem == name
            ]

            if not matches:
                print(f"{name}: image not found")
                continue

            image = cv2.imread(str(matches[0]))

        faces = detector.detect(image)

        if len(faces) != 1:
            print(f"{name}: expected one face, found {len(faces)}")
            continue

        embedding = encoder.encode(
            image,
            faces[0]["face_data"]
        )

        result = matcher.match(embedding)

        print(f"\nExpected: {name}")
        print(f"Predicted: {result['identity']}")
        print(f"Similarity: {result['similarity']:.4f}")

        if result["identity"] == name:
            print("Result: PASS")
        else:
            print("Result: CHECK")


if __name__ == "__main__":
    main()
