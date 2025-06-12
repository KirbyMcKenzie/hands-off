import cv2
import mediapipe as mp
import subprocess
import time

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Threshold distance between hand landmarks and face/neck landmarks to consider "touching"
TOUCH_THRESHOLD = 0.1  # Adjust as needed (normalized distance)

# Minimum duration (in seconds) hands must stay near face/neck to trigger notification
MIN_TOUCH_DURATION = 2.0


def send_notification(title, message):
    subprocess.run(["afplay", f"/System/Library/Sounds/Glass.aiff"])
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}"'
    ])

    print(f"🔔 Face touched for more than {MIN_TOUCH_DURATION} seconds")

def landmarks_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

def is_hand_near_face(hand_landmarks, face_landmarks):
    # Check hand landmarks against key face landmarks and neck area (neck approx from face mesh points)
    # Example face landmarks indices for chin and neck-ish areas:
    # Chin: 152, Neck area approximation: 234, 454 (left and right jawline points)
    face_points_indices = [152, 234, 454]

    for h_landmark in hand_landmarks.landmark:
        for idx in face_points_indices:
            f_landmark = face_landmarks.landmark[idx]
            dist = landmarks_distance(h_landmark, f_landmark)
            if dist < TOUCH_THRESHOLD:
                print(f"🔍 Hand landmark near face: {h_landmark.x:.2f}, {h_landmark.y:.2f} (dist {dist:.3f})")
                return True
    return False

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh:

        touch_start_time = None
        notified = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for natural selfie view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face mesh
            face_results = face_mesh.process(rgb_frame)
            if not face_results.multi_face_landmarks:
                # No face detected, reset timer and flags
                touch_start_time = None
                notified = False
                # Uncomment if you want to see the video:
                # cv2.imshow('Face Touch Alert', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                continue

            face_landmarks = face_results.multi_face_landmarks[0]

            # Process hands
            hand_results = hands.process(rgb_frame)
            hand_near_face = False

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if is_hand_near_face(hand_landmarks, face_landmarks):
                        hand_near_face = True
                        break

            # Timing logic
            if hand_near_face:
                if touch_start_time is None:
                    touch_start_time = time.time()
                    notified = False
                else:
                    elapsed = time.time() - touch_start_time
                    if elapsed >= MIN_TOUCH_DURATION and not notified:
                        send_notification("Stop Touching Your Face", "Please keep your hands away!")
                        notified = True
            else:
                # Hand moved away, reset timer and notification flag
                touch_start_time = None
                notified = False

            # Optional: display video feed window if you want (disable to run headless)
            # cv2.imshow('Face Touch Alert', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
