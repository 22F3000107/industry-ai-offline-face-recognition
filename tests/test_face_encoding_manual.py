
import cv2
from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder


def main():
    image_path = "data/group_photos/test.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    detector = FaceDetector()
    encoder = FaceEncoder()

    faces = detector.detect(image)

    print("\nFace Embedding Results")
    print("----------------------")
    print("Detected faces:", len(faces))

    for index, face in enumerate(faces, start=1):
        embedding = encoder.encode(
            image,
            face["face_data"],
        )

        print(f"\nFace {index}")
        print("Embedding shape:", embedding.shape)
        print("Embedding dimensions:", len(embedding))
        print("Embedding norm:", round(
            float((embedding ** 2).sum() ** 0.5), 4
        ))

    print("\nAll face embeddings generated successfully!")


if __name__ == "__main__":
    main()
