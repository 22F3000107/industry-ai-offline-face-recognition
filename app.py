
import streamlit as st
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.database import ReferenceDatabase
from src.face_matcher import FaceMatcher


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Who Is in the Photo?",
    page_icon="👤",
    layout="wide",
)

MODEL_DIR = Path("models")

DETECTION_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

DEFAULT_THRESHOLD = 0.363


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    detector = FaceDetector(str(DETECTION_MODEL))
    encoder = FaceEncoder(str(RECOGNITION_MODEL))
    return detector, encoder


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def read_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR,
    )


def build_reference_database(folder_path, detector, encoder):
    database = ReferenceDatabase(detector, encoder)
    report = database.build(folder_path)
    return database, report



def annotate_image(image_bgr, results):
    """
    Draw readable face boxes and non-overlapping labels.
    Green = recognized, orange = unknown.
    """

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    output = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(output)

    image_width, image_height = output.size

    # Adapt font size to image resolution.
    font_size = max(
        12,
        min(20, image_width // 120),
    )

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc",
            font_size,
        )
    except OSError:
        font = ImageFont.load_default()

    box_width = max(2, image_width // 1000)
    padding = max(4, font_size // 4)
    gap = max(4, font_size // 3)

    occupied_labels = []

    # Draw larger faces first.
    sorted_results = sorted(
        results,
        key=lambda item: item["bbox"][2] * item["bbox"][3],
        reverse=True,
    )

    for result in sorted_results:
        x, y, w, h = result["bbox"]
        name = result["identity"]
        similarity = result["similarity"]

        if name != "Unknown":
            color = (0, 160, 90)
            label = f"{name} | {similarity:.2f}"
        else:
            color = (225, 135, 25)
            label = f"Unknown | {similarity:.2f}"

        # Face bounding box.
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=color,
            width=box_width,
        )

        # Calculate label dimensions.
        text_box = draw.textbbox(
            (0, 0),
            label,
            font=font,
        )

        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        label_width = text_width + 2 * padding
        label_height = text_height + 2 * padding

        # Candidate positions: above, below, and inside face box.
        candidates = [
            (x, y - label_height - gap),
            (x, y + h + gap),
            (x, y + gap),
        ]

        # Try additional positions if nearby labels overlap.
        for offset in range(1, 8):
            candidates.append(
                (x, y - label_height - gap - offset * (label_height + gap))
            )
            candidates.append(
                (x, y + h + gap + offset * (label_height + gap))
            )

        selected_position = None

        for candidate_x, candidate_y in candidates:
            candidate_x = max(
                0,
                min(candidate_x, image_width - label_width),
            )

            candidate_y = max(
                0,
                min(candidate_y, image_height - label_height),
            )

            candidate_box = (
                candidate_x,
                candidate_y,
                candidate_x + label_width,
                candidate_y + label_height,
            )

            overlaps = any(
                not (
                    candidate_box[2] <= existing[0]
                    or candidate_box[0] >= existing[2]
                    or candidate_box[3] <= existing[1]
                    or candidate_box[1] >= existing[3]
                )
                for existing in occupied_labels
            )

            if not overlaps:
                selected_position = (
                    candidate_x,
                    candidate_y,
                )
                occupied_labels.append(candidate_box)
                break

        if selected_position is None:
            # Fallback: place label inside its face bounding box.
            selected_position = (
                max(0, min(x, image_width - label_width)),
                max(0, min(y + gap, image_height - label_height)),
            )

        label_x, label_y = selected_position

        # Label background.
        draw.rounded_rectangle(
            [
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height,
            ],
            radius=4,
            fill=color,
        )

        # White label text.
        draw.text(
            (
                label_x + padding,
                label_y + padding - text_box[1],
            ),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    return output



def recognize_faces(image, detector, encoder, matcher):
    faces = detector.detect(image)
    results = []

    for face in faces:
        embedding = encoder.encode(
            image,
            face["face_data"],
        )

        prediction = matcher.match(embedding)

        results.append({
            "bbox": face["bbox"],
            "identity": prediction["identity"],
            "similarity": prediction["similarity"],
            "detection_confidence": face["confidence"],
        })

    return results


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">👤 Who Is in the Photo?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Offline face recognition powered by OpenCV YuNet and SFace'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Privacy-first processing: photographs and face embeddings "
    "are processed locally. No online recognition APIs are used."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Recognition threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Higher values make unknown-person rejection stricter.",
    )

    st.caption(
        "Initial threshold: 0.363. "
        "Validate it using representative data before deployment."
    )

    st.divider()

    st.markdown("### Recognition models")
    st.write("**Detection:** YuNet")
    st.write("**Recognition:** SFace")
    st.write("**Processing:** Local / Offline")


# --------------------------------------------------
# INITIALIZE MODELS
# --------------------------------------------------

try:
    detector, encoder = load_models()
except Exception as error:
    st.error(f"Unable to load recognition models: {error}")
    st.stop()


# --------------------------------------------------
# REFERENCE ENROLLMENT
# --------------------------------------------------

st.header("1. Reference Face Enrollment")

st.write(
    "Choose a local folder containing individually named "
    "reference photographs."
)

reference_folder = st.text_input(
    "Reference folder path",
    value="data/reference_faces",
    help="Example: data/reference_faces",
)

if st.button("Build Reference Database", type="primary"):
    folder = Path(reference_folder).expanduser()

    with st.spinner("Processing reference photographs..."):
        try:
            database, report = build_reference_database(
                folder,
                detector,
                encoder,
            )

            st.session_state["identities"] = database.identities
            st.session_state["reference_report"] = report
            st.session_state["reference_folder"] = str(folder)

        except Exception as error:
            st.error(f"Reference enrollment failed: {error}")


if "reference_report" in st.session_state:
    report = st.session_state["reference_report"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Total images", report["total_images"])
    col2.metric("Enrolled identities", report["successful"])
    col3.metric("Skipped images", len(report["errors"]))

    if report["errors"]:
        with st.expander("View skipped reference images"):
            for error in report["errors"]:
                st.warning(error)

    if report["successful"] > 0:
        st.success(
            f"{report['successful']} identities enrolled successfully."
        )

        if st.button("Save Embedding Database"):
            database = ReferenceDatabase(detector, encoder)
            database.identities = st.session_state["identities"]

            output_path = Path("data/reference_embeddings.npz")
            database.save(output_path)

            st.success(f"Database saved to {output_path}")


# --------------------------------------------------
# GROUP PHOTO
# --------------------------------------------------

st.divider()

st.header("2. Upload Group Photograph")

uploaded_file = st.file_uploader(
    "Choose a group photograph",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = read_uploaded_image(uploaded_file)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded group photograph",
        use_container_width=True,
    )

    if "identities" not in st.session_state:
        st.warning("Build the reference database first.")

    else:
        if st.button("Recognize People", type="primary"):
            with st.spinner("Detecting and recognizing faces..."):
                try:
                    matcher = FaceMatcher(
                        st.session_state["identities"],
                        threshold=threshold,
                    )

                    results = recognize_faces(
                        image,
                        detector,
                        encoder,
                        matcher,
                    )

                    st.session_state["recognition_results"] = results
                    st.session_state["annotated_image"] = annotate_image(
                        image,
                        results,
                    )

                except Exception as error:
                    st.error(f"Recognition failed: {error}")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if "recognition_results" in st.session_state:
    results = st.session_state["recognition_results"]

    st.divider()
    st.header("3. Recognition Results")

    recognized = [
        result for result in results
        if result["identity"] != "Unknown"
    ]

    unknown = [
        result for result in results
        if result["identity"] == "Unknown"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric("Faces detected", len(results))
    col2.metric("Recognized", len(recognized))
    col3.metric("Unknown", len(unknown))

    st.subheader("Annotated Photograph")

    st.image(
        st.session_state["annotated_image"],
        use_container_width=True,
    )

    st.subheader("Individual Recognition Results")

    if results:
        table_data = []

        for index, result in enumerate(results, start=1):
            table_data.append({
                "Face": f"Person {index}",
                "Identity": result["identity"],
                "Similarity": round(result["similarity"], 4),
                "Detection confidence": round(
                    result["detection_confidence"], 4
                ),
                "Bounding box": str(result["bbox"]),
            })

        dataframe = pd.DataFrame(table_data)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = dataframe.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Recognition Results CSV",
            data=csv_data,
            file_name="recognition_results.csv",
            mime="text/csv",
        )

    else:
        st.warning("No faces detected in the group photograph.")
