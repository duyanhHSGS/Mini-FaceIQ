from pathlib import Path

import cv2
import mediapipe as mp

_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "features_rating_bundle"
    / "face_landmarker.task"
)
_face_landmarker = None


def get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        if not _MODEL_PATH.is_file():
            raise FileNotFoundError(f"MediaPipe task model not found: {_MODEL_PATH}")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        _face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def get_landmarks_mp(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = get_face_landmarker().detect(image)
    if not result.face_landmarks:
        return None

    h, w = img_bgr.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]
