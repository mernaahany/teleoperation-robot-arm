# 🤖 Teleoperation Robot Arm using AI, Hand Detection & ROS ✋

This project enables real-time teleoperation of a 6-DOF robotic arm using **AI-based hand gesture recognition**, computer vision, and **ROS-based control architecture**. Human hand motion is translated into joint angles and sent to a Raspberry Pi, which controls the robot through ROS.

---

## 🚀 Project Features

- 🎯 Real-time hand tracking using **MediaPipe**
- ✋ Grip detection (open/close) using thumb-index distance
- 🔄 Hand orientation and position mapped to joint angles
- 🔧 Control of 6-DOF robotic arm via socket + ROS
- ⚡ Live visual feedback with OpenCV
- 📡 Bluetooth-based wheel control using Arduino + HC-05
- 💻 Python-based interface + Raspberry Pi + ROS

---

## 🛠️ Tools & Technologies

| Tool / Library       | Purpose                            |
|----------------------|-------------------------------------|
| `MediaPipe`          | Hand landmark detection (21 points) |
| `OpenCV`             | Camera input & visual overlay       |
| `NumPy`              | Math operations & angle calculation |
| `Python socket`      | Communication with Raspberry Pi     |
| `ROS`                | Robot control & message transport   |
| `Arduino + HC-05`    | Wireless wheel base control         |

---

## 🧠 System Overview

1. **MediaPipe** detects 21 hand landmarks in real time.
2. The wrist coordinates define the hand's X, Y, Z position.
3. Orientation is extracted based on the direction from wrist to middle finger.
4. Grip detection is based on thumb-index distance.
5. A mapping function converts all parameters to joint angles:
   - Base rotation
   - Shoulder lift
   - Elbow angle
   - Wrist pitch & roll
   - Gripper open/close
6. The angles are sent to a Raspberry Pi via socket.
7. The Raspberry Pi forwards them via **ROS topics** to the robot nodes controlling the motors.

---

## 📷 Interface Preview

To be added: Images and/or video of the system in action (hand tracking + robot response).

You can include:
- Screenshots from OpenCV showing landmark detection
- Real-life photo or GIF of the robot moving
- Architecture diagram of your system

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Contents:
```
opencv-python
mediapipe
numpy
```

---

## 🧪 How to Run

### PC Side (Hand Detection + Socket Sender)

1. Connect camera and ensure Python environment is set up.
2. Edit IP in `hand_to_robot_control.py`:
```python
raspberry_ip = 'YOUR_PI_IP'
```
3. Run the script:
```bash
python hand_to_robot_control.py
```

### Raspberry Pi Side (Receiver + ROS Publisher)

1. Set up a ROS workspace and custom node that subscribes to joint angles.
2. Start the ROS core:
```bash
roscore
```
3. Run your listener node:
```bash
rosrun your_pkg angle_receiver.py
```
(Your node should decode the socket input and publish to ROS topics.)

---

## 🛞 Wheel Control (Arduino + Bluetooth)

- Separate control for the robot base via Bluetooth module HC-05.
- Arduino receives single-character commands like:
  - `'F'` → forward
  - `'B'` → backward
  - `'L'` → turn left
  - `'R'` → turn right
  - `'S'` → stop

---

## 💡 Future Improvements

- Switch to **ROS 2** or integrate full ROS node on PC.
- Add object detection for target grasping.
- Add IK solver with real calibration.
- Support multiple hand gestures and users.

---

## 👤 Author

**Merna Hany**  
Mechatronics Engineer | AI & Robotics Enthusiast  
📧 mernahany00@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mernahanylin/)  
🔗 GitHub: [github.com/merna-hany](https://github.com/merna-hany)

---

## 📜 License

MIT License – feel free to use, share, or modify.