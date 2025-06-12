import cv2
import mediapipe as mp
import subprocess
import random
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

alert_messages = [
    "Hands near your face! Please stop.",
    "Avoid touching your face or neck.",
    "Reminder: Keep your hands away!",
    "Hands off your face for your health!",
    "Be mindful: hands near face detected!",
    "Stop! Don't touch your face or neck!"
]

def send_notification(title, message):
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}"'
    ])
    subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"])

def hand_near_face_or_neck(hand_landmarks, face_landmarks, neck_landmarks, threshold=0.15):
    if not hand_landmarks or not face_landmarks:
        return False

    # Check if any hand landmark is close to any face or neck landmark (in normalized coordinates)
    for hlm in hand_landmarks.landmark:
        for flm in face_landmarks.landmark:
            dx = hlm.x - flm.x
            dy = hlm.y - flm.y
            dist = (dx*dx + dy*dy)**0.5
            if dist < threshold:
                return True
        for nlk in neck_landmarks:
            dx = hlm.x - nlk[0]
            dy = hlm.y - nlk[1]
            dist = (dx*dx + dy*dy)**0.5
            if dist < threshold:
                return True
    return False

def main():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        alert_cooldown = 5  # seconds
        last_alert_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            # Extract landmarks
            right_hand = results.right_hand_landmarks
            left_hand = results.left_hand_landmarks
            face_landmarks = results.face_landmarks

            # Approximate neck points based on shoulders (pose landmarks)
            neck_landmarks = []
            if results.pose_landmarks:
                # Using shoulders as proxy for neck
                left_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                # Midpoint between shoulders as neck approx
                neck_x = (left_shoulder.x + right_shoulder.x) / 2
                neck_y = (left_shoulder.y + right_shoulder.y) / 2
                neck_landmarks.append((neck_x, neck_y))

            # Check hands near face or neck
            hands_near = False
            if right_hand and face_landmarks and neck_landmarks:
                hands_near = hand_near_face_or_neck(right_hand, face_landmarks, neck_landmarks)
            if not hands_near and left_hand and face_landmarks and neck_landmarks:
                hands_near = hand_near_face_or_neck(left_hand, face_landmarks, neck_landmarks)

            # Alert with cooldown
            if hands_near:
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    alert_msg = random.choice(alert_messages)
                    send_notification("Warning", alert_msg)
                    print("🔔 Notification sent:", alert_msg)
                    last_alert_time = current_time

            # No video display, so just keep looping

            # Small sleep to reduce CPU
            time.sleep(0.01)

if __name__ == "__main__":
    main()
