import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print ("Kamera tidak dapat diakses")
        break

    # Efek cermin & Ukuran layar
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Konversi warna & Proses deteksi AI
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Membaca & Menghitung koordinat ujung Telunjuk
    if result.multi_hand_landmarks :
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            # Visualisasi di layar
            cv2.circle(
                frame,
                (cx, cy),
                12,
                (0, 0, 255),
                cv2.FILLED
            )

            cv2.putText(
                frame,
                f"Telunjuk: ({cx}, {cy})",
                (cx + 15, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Tampilkan hasil & Tutup program
    cv2.imshow("Hari 3- Tracking Telunjuk", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") :
        break

cap.release()
cv2.destroyAllWindows()

