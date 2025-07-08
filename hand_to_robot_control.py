import cv2
import mediapipe as mp
import numpy as np
import time
import socket  # for communication

#  Socket Communication Setup
raspberry_ip = '192.168.117.88'
port = 8888
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((raspberry_ip, port))

# MediaPipe Setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Control Parameters 
last_send_time = time.time()
delay_between_sends = 0.3  # 300ms
prev_sent_angles = np.zeros(6)
angle_change_threshold = 5.0  # degrees

#  Grip Parameters 
GRIP_DISTANCE_THRESHOLD = 0.05
GRIP_OPEN_ANGLE = 110
GRIP_CLOSE_ANGLE = 90
last_grip_state = False

#  Utility Functions 
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_grip(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = calculate_distance([thumb_tip.x, thumb_tip.y], [index_tip.x, index_tip.y])

    # Visualize grip distance
    thumb_pos = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
    index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
    cv2.line(frame, thumb_pos, index_pos, (0, 255, 255), 2)
    cv2.putText(frame, f"{distance:.2f}",
               ((thumb_pos[0] + index_pos[0])//2, (thumb_pos[1] + index_pos[1])//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return distance < GRIP_DISTANCE_THRESHOLD

def calculate_hand_position(landmarks, frame_shape):
    wrist = landmarks[0]
    x_pos = 2 * (wrist.x - 0.5)
    y_pos = 2 * (0.5 - wrist.y)
    hand_size = calculate_distance([landmarks[0].x, landmarks[0].y],
                                   [landmarks[9].x, landmarks[9].y])
    z_pos = min(max(hand_size * 2, 0), 1)
    return x_pos, y_pos, z_pos

def calculate_hand_orientation(landmarks):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    vec_x = middle_mcp.x - wrist.x
    vec_y = middle_mcp.y - wrist.y
    angle_rad = np.arctan2(vec_y, vec_x)
    angle_deg = np.degrees(angle_rad)
    return max(-90, min(90, angle_deg))

def map_to_robot_angles(x_pos, y_pos, z_pos, orientation, grip_state):
    base_angle = x_pos * 90
    shoulder_angle = 20 + (z_pos * 70)
    elbow_angle = 20 + (y_pos * 70)
    wrist_roll = orientation * 0.8
    wrist_pitch = 45 - (z_pos * 45)
    gripper = GRIP_CLOSE_ANGLE if grip_state else GRIP_OPEN_ANGLE
    return np.array([
        base_angle,
        shoulder_angle,
        elbow_angle,
        wrist_pitch,
        wrist_roll,
        gripper
    ])

#  Main Loop 
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            x_pos, y_pos, z_pos = calculate_hand_position(landmarks, frame.shape)
            orientation = calculate_hand_orientation(landmarks)
            grip_state = detect_grip(landmarks)

            target_angles = map_to_robot_angles(x_pos, y_pos, z_pos, orientation, grip_state)

            angle_diff = np.linalg.norm(target_angles - prev_sent_angles)
            current_time = time.time()

            if (current_time - last_send_time > delay_between_sends and 
                (angle_diff > angle_change_threshold or grip_state != (prev_sent_angles[5] != GRIP_OPEN_ANGLE))):
                
                data_str = ",".join(f"{angle:.2f}" for angle in target_angles)
                client_socket.sendall((data_str + "\n").encode('utf-8'))
                print(f"Sent angles: {data_str}")

                prev_sent_angles = target_angles.copy()
                last_send_time = current_time

            # Visual feedback
            grip_color = (0, 0, 255) if grip_state else (0, 255, 0)
            cv2.circle(frame, (50, 50), 20, grip_color, -1)
            cv2.putText(frame, f"Grip: {GRIP_CLOSE_ANGLE}° (CLOSED)" if grip_state else f"Grip: {GRIP_OPEN_ANGLE}° (OPEN)", 
                       (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, grip_color, 2)
            cv2.putText(frame, f"X: {x_pos:.2f} Y: {y_pos:.2f} Z: {z_pos:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Orientation: {orientation:.1f}°", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Robot Arm Control (Grip: 110° Open / 90° Closed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Cleanup 
cap.release()
cv2.destroyAllWindows()
client_socket.close()
