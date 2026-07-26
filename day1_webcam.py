import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print ("Gambar dari kamera tidak terbaca.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.rectangle(
        frame, 
        (300, 140),
        (620, 600),
        (0, 255, 0),
        8)
    cv2.circle(
        frame,
        (460, 370),
        100,
        (255, 0, 0),
        8)
    cv2.putText(
        frame,
        "Kamera saya",
        (200, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3)
    
    cv2.imshow("Kamera saya", frame)

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
