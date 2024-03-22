import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Intention: When reading literature, I don't want to hold the mouse and control the computer from a distance to flip
# through documents. Please disable antivirus or protective software before use, otherwise the software will
# intercept the click action, causing the click to fail. Control gestures: 1: Move up -> Any hand's index finger
# pointing up, thumb pointing up 2: Move down -> Any hand's index finger pointing down, thumb pointing down 3: Move
# left -> Any hand's index finger pointing left, thumb pointing left 4: Move right -> Any hand's index finger
# pointing right, thumb pointing right 5: Scroll up -> Any hand's middle finger pointing right 6: Scroll down -> Any
# hand's middle finger pointing left 7: Click -> Any hand's thumb pointing up

# Get the screen width and height
screen_width, screen_height = pyautogui.size()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def identify_which_finger_point(array_landmarks):
    # Calculate the center point of the 21 key nodes
    center_x = 0
    center_y = 0
    for i in range(21):
        center_x += array_landmarks[i, 0]
        center_y += array_landmarks[i, 1]
    center_x = int(center_x / 21)
    center_y = int(center_y / 21)
    cv2.circle(image, (center_x, center_y), 5, (0, 255, 255), cv2.FILLED)

    # The index of the extended finger, 0->thumb, 1->index finger, 2->middle finger, 3->ring finger, 4->little
    # finger, 5->no finger extended
    point_id = 0
    # Maximum distance value from the node to the center point
    dis_from_center_max = 0
    # Distance from the node to the center point
    dis_from_center = 0
    # Determine the distance from each key node to the center point
    for i in range(1, 21):
        dis_from_center = abs(array_landmarks[i, 0] - center_x) + abs(array_landmarks[i, 1] - center_y)
        if dis_from_center > dis_from_center_max:
            dis_from_center_max = dis_from_center
            if i == 4:
                point_id = 0
            elif i == 8:
                point_id = 1
            elif i == 12:
                point_id = 2
            elif i == 16:
                point_id = 3
            elif i == 20:
                point_id = 4
            else:
                point_id = 5
    return point_id


def calculate_angle(x1, y1, x2, y2, x3, y3, x4, y4):
    # Coordinate difference of vector 1
    dx1 = x2 - x1
    dy1 = y2 - y1

    # Coordinate difference of vector 2
    dx2 = x4 - x3
    dy2 = y4 - y3

    # Calculate the dot product of vector 1 and vector 2
    dot_product = dx1 * dx2 + dy1 * dy2

    # Calculate the magnitude of vector 1 and vector 2
    magnitude1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
    magnitude2 = math.sqrt(dx2 ** 2 + dy2 ** 2)

    # Calculate the cosine of the dot product
    cosine = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle using the arccos function
    angle = math.degrees(math.acos(cosine))

    # Calculate the cross product of vector 1 and vector 2
    cross_product = dx1 * dy2 - dx2 * dy1

    # Adjust the angle sign based on the sign of the cross product
    if cross_product < 0:
        angle = -angle

    return angle


# Check if the mouse position is close to the screen edge, return 1 for safe, 0 for unsafe
def is_safe(move_x, move_y):
    lx, ly = pyautogui.position()
    aim_x = lx + move_x
    aim_y = ly + move_y
    if 0 < aim_x < screen_width and 0 < aim_y < screen_height:
        return 1
    else:
        return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Initialize the MediaPipe hand gesture recognition module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)  # Use OpenCV to call the camera, 0 == camera, file path == open video
    if cap is None or not cap.isOpened():
        print("Unable to open camera.")
        exit()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read image from camera")
            break
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Initialize cv2 image
        results = hands.process(img)

        # Create an array of size 21x2 to store the data of the 21 recognized keypoints
        array_landmarks = np.zeros((21, 2))
        # Draw the detected gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    array_landmarks[id, 0] = cx
                    array_landmarks[id, 1] = cy
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # If it's the index finger extended, control mouse movement according to the finger direction
            angle = 45  # The angle between the index finger and the positive direction of the x-axis (-180,180),
            # close to 0 for right, close to 90 for up, close to ±180 for left, close to -90 for down
            finger_id = identify_which_finger_point(array_landmarks)

            # print(finger_id)
            if finger_id == 1:
                angle = calculate_angle(0, 0, 100, 0, array_landmarks[8, 0], array_landmarks[8, 1],
                                        array_landmarks[7, 0], array_landmarks[7, 1])
                if 40 > angle > -40:
                    if is_safe(10, 0):
                        pyautogui.move(10, 0)
                        print("Move right")
                elif 50 < angle < 130:
                    if is_safe(0, -10):
                        pyautogui.move(0, -10)
                        print("Move up")
                elif 140 < angle or angle < -140:
                    if is_safe(-10, 0):
                        pyautogui.move(-10, 0)
                        print("Move left")
                elif -130 < angle < -50:
                    if is_safe(0, 10):
                        pyautogui.move(0, 10)
                        print("Move down")
                else:
                    pass
            elif finger_id == 0:
                angle = calculate_angle(0, 0, 100, 0, array_landmarks[4, 0], array_landmarks[4, 1],
                                        array_landmarks[3, 0], array_landmarks[3, 1])
                if 40 > angle > -40:
                    if is_safe(10, 0):
                        pyautogui.move(10, 0)
                        print("Move right")
                elif 50 < angle < 130:
                    time.sleep(0.01)
                    x, y = pyautogui.position()
                    pyautogui.click(x, y)  # Please disable antivirus or protection software to prevent failure
                    print("Left click")
                elif 140 < angle or angle < -140:
                    if is_safe(-10, 0):
                        pyautogui.move(-10, 0)
                        print("Move left")
                elif -130 < angle < -50:
                    if is_safe(0, 10):
                        pyautogui.move(0, 10)
                        print("Move down")
                else:
                    pass
            elif finger_id == 2:
                angle = calculate_angle(0, 0, 100, 0, array_landmarks[12, 0], array_landmarks[12, 1],
                                        array_landmarks[11, 0], array_landmarks[11, 1])
                if -20 < angle < 20:
                    pyautogui.scroll(5)
                    print("Scroll up")
                elif 160 < angle or angle < -160:
                    pyautogui.scroll(-5)
                    print("Scroll down")
                else:
                    pass

        # Display the image (for debugging purposes, this can be commented out or removed in actual use)
        image = cv2.flip(image, 1)
        cv2.imshow("Hand Gestures", image)
        # cv2.imshow("Image", image)  # CV2 window, showing the video stream captured by the camera
        # cv2.imshow("Image2", img)
        end_time = time.time()
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.waitKey(1)  # Close the window
