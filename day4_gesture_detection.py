import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print ("Kamera tidak dapat diakses")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks :
        for hand_landmarks in result.multi_hand_landmarks:
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            # Koordinat
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Hitung Tengah dan Jarak
            cx, cy = (x1 + x2) // 2, (y1 + y2)// 2
            distance = math.hypot(x2 - x1, y2 - y1)

            # Indicator
            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Logic
            if distance < 30:
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

                cv2.putText(
                    frame,
                    "STATUS: KLIK (PINCH)",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "STATUS: DIAM",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # Tampilkan jarak
            cv2.putText(
                frame,
                f"Jarak: {int(distance)} px",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Gestur Pinch (klik)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
