#!/bin/bash
# Guitar Neck Detection Webcam Launcher

cd /home/shaken/fun_guitar
source venv/bin/activate
QT_QPA_PLATFORM=xcb python3 webcam_guitar_detection.py
