import 'package:flet/flet.dart';
import 'package:flutter/widgets.dart';
import 'face_landmarker_service.dart';

class Extension extends FletExtension {
  @override
  Widget? createWidget(Key? key, Control control) => null;

  @override
  FletService? createService(Control control) {
    if (control.type == 'FletFaceLandmarker') {
      return FaceLandmarkerService(control: control);
    }
    return null;
  }
}
