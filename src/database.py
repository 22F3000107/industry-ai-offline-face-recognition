
from pathlib import Path
import cv2
import numpy as np

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder


class ReferenceDatabase:
    """Build and manage a local database of known face embeddings."""

    def __init__(self, detector=None, encoder=None):
        self.detector = detector or FaceDetector()
        self.encoder = encoder or FaceEncoder()
        self.identities = {}

    def build(self, folder_path):
        """
        Process reference photographs.

        Each image must contain exactly one detectable face.
        The filename (without extension) becomes the identity.
        """

        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            raise ValueError(
                f"Reference folder does not exist: {folder}"
            )

        extensions = {".jpg", ".jpeg", ".png"}
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not image_paths:
            raise ValueError("No reference photographs found.")

        self.identities = {}
        errors = []

        for image_path in image_paths:
            name = image_path.stem.strip()

            if not name:
                errors.append(f"{image_path.name}: empty filename")
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                errors.append(f"{image_path.name}: invalid image")
                continue

            faces = self.detector.detect(image)

            if len(faces) != 1:
                errors.append(
                    f"{image_path.name}: expected 1 face, "
                    f"found {len(faces)}"
                )
                continue

            embedding = self.encoder.encode(
                image,
                faces[0]["face_data"],
            )

            self.identities[name] = embedding

        return {
            "total_images": len(image_paths),
            "successful": len(self.identities),
            "errors": errors,
        }

    def save(self, output_path="data/reference_embeddings.npz"):
        """Save embeddings locally."""

        if not self.identities:
            raise ValueError("No identities available to save.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        names = list(self.identities.keys())
        embeddings = np.stack(
            [self.identities[name] for name in names]
        )

        np.savez_compressed(
            output,
            names=np.asarray(names),
            embeddings=embeddings,
        )

        return output

    def load(self, input_path="data/reference_embeddings.npz"):
        """Load a previously saved local embedding database."""

        data = np.load(input_path, allow_pickle=False)

        names = data["names"].tolist()
        embeddings = data["embeddings"]

        if len(names) != len(embeddings):
            raise ValueError("Invalid embedding database.")

        self.identities = {
            name: embedding
            for name, embedding in zip(names, embeddings)
        }

        return len(self.identities)

