
import cv2

from src.database import ReferenceDatabase
from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_matcher import FaceMatcher


def main():
    # Load the existing reference database.
    database = ReferenceDatabase()
    count = database.load("data/reference_embeddings.npz")

    detector = FaceDetector()
    encoder = FaceEncoder()

    matcher = FaceMatcher(
        database.identities,
        threshold=0.363
    )

    image = cv2.imread("data/group_photos/test.jpg")

    if image is None:
        raise FileNotFoundError(
            "Group photograph not found."
        )

    faces = detector.detect(image)

    print("\nFace Matching Results")
    print("---------------------")
    print("Reference identities:", count)
    print("Detected faces:", len(faces))

    for index, face in enumerate(faces, start=1):
        embedding = encoder.encode(
            image,
            face["face_data"]
        )

        result = matcher.match(embedding)

        print(f"\nFace {index}")
        print("Identity:", result["identity"])
        print("Best candidate:", result["best_match"])
        print("Similarity:", round(result["similarity"], 4))
        print("Threshold:", result["threshold"])


if __name__ == "__main__":
    main()
