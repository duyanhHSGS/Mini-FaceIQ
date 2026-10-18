Pod::Spec.new do |s|
  s.name             = 'flet_face_landmarker'
  s.version          = '0.1.0'
  s.summary          = 'On-device face landmarking for Mini-FaceIQ'
  s.description      = 'Flutter method channel for offline iOS MediaPipe Face Landmarker.'
  s.homepage         = 'https://github.com/google-ai-edge/mediapipe'
  s.license          = { :type => 'MIT' }
  s.author           = { 'Mini-FaceIQ' => 'Mini-FaceIQ' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  # Version 0.10.21 avoids the 0.10.33+ CocoaPods XCFramework regression.
  s.dependency 'MediaPipeTasksVision', '0.10.21'
  s.platform = :ios, '15.0'
  s.swift_version = '5.0'
  s.static_framework = true
end
