import os

import cv2
import mediapipe as mp


ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(ROOT, "face_landmarker.task")
_landmarker = None


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"MediaPipe model not found: {_MODEL_PATH}")
        _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
        )
    return _landmarker


def get_landmarks_mp(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = get_landmarker().detect(mp_img)
    if not result.face_landmarks:
        return None

    h, w = img_bgr.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]
