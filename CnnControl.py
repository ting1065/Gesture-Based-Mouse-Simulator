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
labels = ['down', 'left', 'pause', 'right', 'rock', 'thumb down', 'thumb up', 'up']

# Adding a flag to keep track of pause state
is_paused = False

# record the number of click related gestures recently captured 
thumb_up_count = 0
thumb_down_count = 0
rock_count = 0

# clear the count of click related gestures
def clear_count():
    global thumb_up_count, thumb_down_count, rock_count
    thumb_up_count = 0
    thumb_down_count = 0
    rock_count = 0


def perform_action(gesture_class, confidence=1.0):
    global is_paused, thumb_up_count, thumb_down_count, rock_count
    action_label = labels[gesture_class]
    print(f"Performing action: {action_label}, Confidence: {confidence:.2f}")

    if is_paused and action_label != 'pause':
        return  # If paused, ignore other actions except for "pause" to toggle pause state
    
    # Execute click-related actions based on gesture class
    if action_label == 'thumb up':
        if thumb_up_count == 3: # Perform left click if thumb up gesture is detected 3 times in a row in 0.75 seconds
            pyautogui.click()
            clear_count()
            time.sleep(1)
        else:
            thumb_up_count += 1
            time.sleep(0.25)
    elif action_label == 'thumb down':
        if thumb_down_count == 3: # Perform right click if thumb down gesture is detected 3 times in a row in 0.75 seconds
            pyautogui.click(button='right')
            clear_count() 
            time.sleep(1)
        else:
            thumb_down_count += 1
            time.sleep(0.25)
    elif action_label == 'rock':
        if rock_count == 3: # Perform double left click if rock gesture is detected 3 times in a row in 0.75 seconds
            pyautogui.doubleClick(interval=0.02, button='left')
            clear_count()
            time.sleep(1)
        else:
            rock_count += 1
            time.sleep(0.25)
    else:
        clear_count()

    # Execute other actions based on gesture class
    if action_label == 'up':
        pyautogui.move(0, -10)  # Move cursor up
        # time.sleep(0.1)
    elif action_label == 'down':
        pyautogui.move(0, 10)  # Move cursor down
        # time.sleep(0.1)
    elif action_label == 'left':
        pyautogui.move(-10, 0)  # Move cursor left
        # time.sleep(0.1)
    elif action_label == 'right':
        pyautogui.move(10, 0)  # Move cursor right
        # time.sleep(0.1)
    elif action_label == 'pause':
        is_paused = not is_paused  # Toggle pause state
        print(f"Gesture recognition paused: {is_paused}")
        time.sleep(1)  # Add a delay after toggling pause state


cap = cv2.VideoCapture(1)

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
            if confidence > 0.99:
                perform_action(gesture_class, confidence)

    # Convert image from RGB back to BGR for display
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Tracking', image_bgr)

    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
