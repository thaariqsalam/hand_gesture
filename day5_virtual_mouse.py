import math
import cv2
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Inisialisasi mediapipe dan ukuran monitor asli
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
)

# Ambil resolusi monitor fisik
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variabel exponential smoothing
smooth_factor = 0.2 # Nilai alpha (0.2 - 0.3 recommend)
prev_x, prev_y = 0, 0 # Menyimpan posisi kursor di frame sebelumnya

# Variabel pembantu cegah spam klik
already_clicked = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kamera tidak dapat diakses")
        break

    # Flip dan ambil resolusi webcam
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Konversi RGB dan deteksi AI
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]

            # Ambil koordinat piksel di frame webcam
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)
            x3, y3 = int(middle.x * w), int(middle.y * h)

            # Hitung jarak thumb & index
            distance = math.hypot(x3 - x1, y3 - y1)

            # Petakan koordinat index finger
            target_x = int(index.x * screen_w)
            target_y = int(index.y * screen_h)

            curr_x = prev_x + (target_x - prev_x) * smooth_factor
            curr_y = prev_y + (target_y - prev_y) * smooth_factor

            # Konversi ke integer untuk posisi mouse fisik
            mouse_x, mouse_y = int(curr_x), int(curr_y)

            # Update posisi sebelumnya untuk perulangan berikutnya
            prev_x, prev_y = curr_x, curr_y

            # Logika CLick Mouse
            if distance < 35:
                cv2.circle(
                    frame,
                    ((x1 + x3)// 2, (y1 + y3)// 2),
                    12,
                    (0, 255, 0),
                    cv2.FILLED,
                )

                # Eksekusi klik hanya sekali saat baru menempel
                if not already_clicked:
                    # Fungsi click dengan koordinat langsung
                    pyautogui.click(x = mouse_x, y = mouse_y)
                    already_clicked = True

                cv2.putText(
                    frame,
                    "STATUS: KLIK",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                # Reset status klik jika jari kembali renggang
                already_clicked = False

                # Kursor gerak HANYA jika tidak sedang eksekusi klik
                pyautogui.moveTo(x = mouse_x, y = mouse_y)

                cv2.putText(
                    frame,
                    ("Status: Moving"),
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # Visualisasi titik dan garis di frame
            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 8, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x3, y3), (255, 0, 0), 2)

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()