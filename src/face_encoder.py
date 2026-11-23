
import cv2
import numpy as np
from pathlib import Path


class FaceEncoder:
    """Generate face embeddings using the OpenCV SFace model."""

    def __init__(
        self,
        model_path="models/face_recognition_sface_2021dec.onnx",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SFace model not found: {self.model_path}"
            )

        self.recognizer = cv2.FaceRecognizerSF_create(
            str(self.model_path),
            "",
        )

    def encode(self, image, face_data):
        """
        Align a detected face and generate its embedding.

        Args:
            image: Original BGR image.
            face_data: Original 15-value YuNet detection.

        Returns:
            Normalized face embedding as a NumPy array.
        """

        if image is None:
            raise ValueError("Input image is empty.")

        face = np.asarray(
            face_data,
            dtype=np.float32,
        ).reshape(1, 15)

        aligned_face = self.recognizer.alignCrop(
            image,
            face,
        )

        embedding = self.recognizer.feature(
            aligned_face
        )

        embedding = embedding.reshape(-1).astype(
            np.float32
        )

        embedding = embedding / (
            np.linalg.norm(embedding) + 1e-12
        )

        return embedding
