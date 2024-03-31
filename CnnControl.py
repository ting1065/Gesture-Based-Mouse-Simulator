import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui
import time

# Load the pre-trained model
model = load_model('hand_gesture_model.h5')

# Initialize MediaPipe hand gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Tool for drawing hand keypoints

# Define gesture category labels
labels = ['down', 'left', 'right', 'thumb down', 'thumb up', 'up']


def perform_action(gesture_class, confidence=1.0):
    """Perform actions based on gesture class and confidence."""
    action_label = labels[gesture_class]
    print(f"Performing action: {action_label}, Confidence: {confidence:.2f}")

    # Execute corresponding actions based on gesture class
    if action_label == 'up':
        pyautogui.move(0, -10)  # Move cursor up
    elif action_label == 'down':
        pyautogui.move(0, 10)  # Move cursor down
    elif action_label == 'left':
        pyautogui.move(-10, 0)  # Move cursor left
    elif action_label == 'right':
        pyautogui.move(10, 0)  # Move cursor right
    elif action_label == 'thumb up':
        time.sleep(0.1)
        pyautogui.click()  # Perform left click
    elif action_label == 'thumb down':
        time.sleep(0.1)
        pyautogui.click(button='right')  # Perform right click


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand keypoints
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
            keypoints = keypoints.reshape(-1, 21, 2, 1)

            # Predict gesture using the model
            prediction = model.predict(keypoints)
            gesture_class = np.argmax(prediction)
            confidence = np.max(prediction)  # Get prediction confidence

            # Display gesture and confidence on the image, changing font color to red for visibility
            gesture_text = f"Gesture: {labels[gesture_class]}, Confidence: {confidence:.2f}"
            cv2.putText(image, gesture_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # If confidence is above a certain threshold, perform the corresponding action
            if confidence > 0.5:
                perform_action(gesture_class, confidence)

    # Convert image from RGB back to BGR for display
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Tracking', image_bgr)

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
