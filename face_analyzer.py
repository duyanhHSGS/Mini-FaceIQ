import cv2
import mediapipe as mp


_face_mesh = None


def get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _face_mesh


def get_landmarks_mp(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = get_face_mesh().process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    h, w = img_bgr.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark

    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
