import 'package:flet/flet.dart';
import 'package:flutter/material.dart';

class FletFaceLandmarkerControl extends StatelessWidget {
  final Control control;

  const FletFaceLandmarkerControl({super.key, required this.control});

  @override
  Widget build(BuildContext context) {
    String text = control.getString("value", "")!;
    Widget myControl = Text(text);

    return LayoutControl(control: control, child: myControl);
  }
}
