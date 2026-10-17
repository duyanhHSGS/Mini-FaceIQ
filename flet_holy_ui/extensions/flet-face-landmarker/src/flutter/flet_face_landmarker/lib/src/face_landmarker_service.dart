import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flet/flet.dart';

/// Flet Python -> Flutter -> native Android MediaPipe Tasks bridge.
class FaceLandmarkerService extends FletService {
  FaceLandmarkerService({required super.control});

  static const _channel = MethodChannel('dev.minifaceiq/face_landmarker');
  bool _initialized = false;

  @override
  void init() {
    super.init();
    control.addInvokeMethodListener(_invoke);
  }

  Future<dynamic> _invoke(String name, dynamic args) async {
    if (name != 'detect') {
      throw Exception('Unknown face landmarker method: $name');
    }
    if (defaultTargetPlatform != TargetPlatform.android) {
      throw UnsupportedError(
        'Native Face Landmarker currently supports Android only',
      );
    }
    final image = base64Decode(args['image'] as String);
    final model = !_initialized ? base64Decode(args['model'] as String) : null;
    final response = await _channel.invokeListMethod<dynamic>('detect', {
      'image': image,
      if (model != null) 'model': model,
    });
    _initialized = true;
    return response ?? <dynamic>[];
  }

  @override
  void dispose() {
    control.removeInvokeMethodListener(_invoke);
    super.dispose();
  }
}
