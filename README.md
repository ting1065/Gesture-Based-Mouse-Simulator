# Gesture-Based Mouse Simulator for Accessibility and Ergonomics (GMSAE)
## Introduction
The GMSAE project aims to develop an application that utilizes MediaPipe combined with CNN model training, to identify and recognize a user’s hand gestures through a computer webcam, and execute corresponding operations of a mouse in the computer.
  
This application is designed to assist users who experience hand tremors and are unable to precisely control a mouse, such as individuals with Parkinson's disease or other reasons and to offer an ergonomic alternative to prevent repetitive strain injuries such as mouse arm syndrome.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Demo](#demo)
- [Dataset for training](#dataset-for-training)

## Requirements

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

## Usage

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

Demo video at [YouTube](https://youtu.be/DVWB55--fL4) showing using hand gestures to open a document and play Minesweeper.

## Dataset for training

Hand gesture videos: https://drive.google.com/drive/folders/1JxRKFOqm2DrPmGaTbgrSw-gwwcGvdv-M?usp=sharing 








