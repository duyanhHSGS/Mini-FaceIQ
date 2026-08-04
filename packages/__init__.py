__all__ = [
    "analyze_features_from_upload",
    "detect_front_landmarks_from_upload",
    "detect_side_landmarks_from_upload",
]


def analyze_features_from_upload(file_storage, suffix):
    from .adapter_features import analyze_features_from_upload as analyze

    return analyze(file_storage, suffix)


def detect_front_landmarks_from_upload(file_storage):
    from .adapter_front import detect_front_landmarks_from_upload as detect

    return detect(file_storage)


def detect_side_landmarks_from_upload(file_storage):
    from .adapter_side import detect_side_landmarks_from_upload as detect

    return detect(file_storage)

