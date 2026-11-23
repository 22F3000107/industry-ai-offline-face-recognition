
import cv2
from src.face_detector import FaceDetector


def main():
    image_path = "data/group_photos/test.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    detector = FaceDetector()

    faces = detector.detect(image)

    print("\nFace Detection Results")
    print("----------------------")
    print("Image dimensions:", image.shape)
    print("Faces detected:", len(faces))

    for index, face in enumerate(faces, start=1):
        print(f"\nFace {index}")
        print("Bounding box:", face["bbox"])
        print("Confidence:", round(face["confidence"], 4))

    # Draw bounding boxes on a copy of the image.
    output = image.copy()

    for index, face in enumerate(faces, start=1):
        x, y, w, h = face["bbox"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"Face {index}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    from pathlib import Path

    output_path = Path("outputs/face_detection_result.jpg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), output)

    if success:
        print("\nOutput saved successfully:", output_path)
    else:
       raise RuntimeError("Failed to save detection result.")


if __name__ == "__main__":
    main()
