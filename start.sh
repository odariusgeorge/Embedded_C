#!/bin/sh

sudo -s
. ~/etc/.profile 
workon cv
python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1
