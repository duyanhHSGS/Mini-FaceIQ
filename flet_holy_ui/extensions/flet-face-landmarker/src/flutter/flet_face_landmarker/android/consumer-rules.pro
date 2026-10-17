# Preserve protobuf-lite reflection metadata used by MediaPipe Tasks.
-keep class * extends com.google.protobuf.GeneratedMessageLite { *; }

# Preserve MediaPipe generated protobuf message classes and fields.
-keep class com.google.mediapipe.tasks.core.proto.** { *; }

# Preserve MediaPipe framework callbacks reached from native JNI.
-keep public interface com.google.mediapipe.framework.* {
    public *;
}

# Preserve MediaPipe JNI-facing classes and methods (release-mode R8).
-keep class com.google.mediapipe.** { *; }
