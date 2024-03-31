# GMSAE
Hand gesutre videos: https://drive.google.com/drive/folders/1JxRKFOqm2DrPmGaTbgrSw-gwwcGvdv-M?usp=sharing 

ExtractData: Extract mediapipe data to create a dataset

CnnTrain: Train CNN model

CnnControl: Use traned model to control the computer

Verified Windows Library Versions:
Python                    3.8.19
tensorflow                2.7.0
mediapipe                 0.8.11
numpy                     1.24.4 
pyautogui                 0.9.54
opencv-contrib-python     4.9.0.80
pandas                    2.0.3
scikit-learn              1.3.2
keras                     2.7.0

Verified Macos M1 Library Versions:
Python                       3.8.18
tensorflow                   2.12.0
mediapipe                    0.9.1.0
keras                        2.12.0
numpy                        1.23.5
PyAutoGUI                    0.9.54

Fix Intel MKL FATAL ERROR: Cannot load libmkl_core.dylib. while running pyspark in MacOs M1:
conda install -c anaconda mkl
