# Gesture-Based Mouse Simulator
## Introduction
The project aims to develop an application that enables computer users to remotely control the cursor using hand gestures, utilizing CNN(Convolutional Neural Network), OpenCV, TensorFlow, Keras, Scikit-learn, MediaPipe and PyAutoGUI.

The functionality to control mouse movements in four directions, single left-click, double left-click, and single right-click has successfully been implemented.

## Table of Contents

- [Configuration](#Configuration)
- [Core Files](#Core-Files)
- [Demo](#Demo)
- [Dataset](#dataset)

## Configuration

Verified Windows Library Versions:
```bash
python==3.11
tensorflow==2.12.0
mediapipe==0.10.11
numpy==1.24.4 
pyautogui==0.9.54
opencv-contrib-python==4.9.0.80
opencv-python==4.9.0.80
pandas==2.2.2
scikit-learn==1.3.2
keras==2.12.0
```

Verified Macos M1 Library Versions:
```bash
python==3.8.18
tensorflow==2.12.0
mediapipe==0.9.1.0
keras==2.12.0
numpy==1.23.5
PyAutoGUI==0.9.54
opencv-contrib-python==4.9.0.80
opencv-python-headless==4.9.0.80
```

Note: to fix Intel MKL FATAL ERROR: Cannot load libmkl_core.dylib. while running pyspark in MacOs M1:
```bash
conda install -c anaconda mkl
```

## Core Files

- Extract data from videos of different hand gestures using MediaPipe to create a dataset:

   ```bash
   python ExtractData.py
   ```

- Train CNN model:
   ```bash
   python CnnTrain.py
   ```

- Use the our trained model to detect hand gestures and control mouse:
   ```bash
   python CnnControl.py
   ```

## Demo

Demo video at [YouTube](https://youtu.be/DVWB55--fL4) showing using hand gestures to open document and play Minesweeper.

## Dataset

Hand gesture videos: [Google Drive](https://drive.google.com/drive/folders/1JxRKFOqm2DrPmGaTbgrSw-gwwcGvdv-M?usp=sharing)