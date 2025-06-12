import cv2
import mediapipe as mp
import subprocess
import time
import random

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

PROXIMITY_THRESHOLD = 0.1
hand_start_time = None
notification_stage = 0
ANGRY_SOUND = "Funk"

NOTIFICATION_MESSAGES_MILD = [
    "Hands off your face!",
    "Stop touching your face!",
    "Keep those hands away!",
    "No face touching!",
]

NOTIFICATION_MESSAGES_ANGRY = [
    "Seriously, move your hand!",
    "Enough! Stop touching your face!",
    "You're doing it again!",
    "Cut it out now!",
]

MEAN_TERMINAL_MESSAGES = [
    "Stop touching your face",
    "Enough warnings, stop now",
    "Stop touching your face now",
    "Thats really gross, stop it",
]

def send_notification(title, message):
    print(f"[NOTIFY] {message}")
    subprocess.run(["afplay", f"/System/Library/Sounds/{ANGRY_SOUND}.aiff"])
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}"'
    ])

def show_dialog(message):
    print("[DIALOG] Showing modal dialog.")
    subprocess.run([
        "osascript", "-e",
        f'display dialog "{message}" buttons {{\"OK\"}} default button 1 with icon caution'
    ])

def say_mean_message():
    phrase = random.choice(MEAN_TERMINAL_MESSAGES)
    print(f"[SAY] {phrase}")
    subprocess.run(["say", "-v", "Samantha", phrase])

def hand_near_landmark(hand_landmark, face_landmark):
    dx = hand_landmark.x - face_landmark.x
    dy = hand_landmark.y - face_landmark.y
    dz = hand_landmark.z - face_landmark.z
    distance = (dx*dx + dy*dy + dz*dz)**0.5
    return distance < PROXIMITY_THRESHOLD

def detect_hand_near_face_or_neck(hand_landmarks, face_landmarks):
    if not hand_landmarks or not face_landmarks:
        return False

    face_points_to_check = [
        face_landmarks.landmark[1],   # Nose
        face_landmarks.landmark[152], # Chin/Neck base
    ]

    for hand_lm in hand_landmarks.landmark:
        for face_lm in face_points_to_check:
            if hand_near_landmark(hand_lm, face_lm):
                print(f"[DETECT] Hand near face at ({hand_lm.x:.2f}, {hand_lm.y:.2f})")
                return True
    return False

def reset_state():
    global hand_start_time, notification_stage
    if hand_start_time or notification_stage > 0:
        print("[STATE] Hand removed. Resetting.")
    hand_start_time = None
    notification_stage = 0

def main():
    global hand_start_time, notification_stage

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Monitoring for face touching...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            hand_detected = False

            if hands_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        if detect_hand_near_face_or_neck(hand_landmarks, face_landmarks):
                            hand_detected = True
                            break
                    if hand_detected:
                        break

            now = time.time()

            if hand_detected:
                if hand_start_time is None:
                    hand_start_time = now
                    print("[STATE] Started timing contact.")

                elapsed = now - hand_start_time
                print(f"[STATE] Hand on face for {elapsed:.2f}s")

                if elapsed >= 3 and notification_stage == 0:
                    msg = random.choice(NOTIFICATION_MESSAGES_MILD)
                    send_notification("Hands off!", msg)
                    notification_stage = 1
                    print("[NOTIFY] First notification sent.")

                elif elapsed >= 6 and notification_stage == 1:
                    msg = random.choice(NOTIFICATION_MESSAGES_ANGRY)
                    send_notification("Warning!", msg)
                    notification_stage = 2
                    print("[NOTIFY] Second, angrier notification sent.")

                elif elapsed >= 10 and notification_stage == 2:
                    say_mean_message()
                    show_dialog("REMOVE YOUR HAND FROM YOUR FACE!")
                    print("[DIALOG] Blocking dialog shown.")
                    reset_state()
            else:
                reset_state()

    except KeyboardInterrupt:
        print("\n[EXIT] Exiting.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
