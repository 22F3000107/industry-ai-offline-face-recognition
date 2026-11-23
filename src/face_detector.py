
import cv2
from pathlib import Path


class FaceDetector:
    """Detect faces in images using the OpenCV YuNet model."""

    def __init__(
        self,
        model_path="models/face_detection_yunet_2023mar.onnx",
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=5000,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {self.model_path}"
            )

        self.detector = cv2.FaceDetectorYN_create(
            str(self.model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, image):
        """
        Detect faces in a BGR image.

        Returns:
            A list of dictionaries containing bounding boxes
            and detection confidence scores.
        """

        if image is None:
            raise ValueError("Input image is empty or invalid.")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image.")

        height, width = image.shape[:2]

        # Update detector dimensions to match the input image.
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        results = []

        if faces is None:
            return results

        for face in faces:
            x, y, w, h = face[:4]

            results.append({
                "bbox": (
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                ),
                "confidence": float(face[-1]),
                "face_data": face.tolist(),
            })

        return results
