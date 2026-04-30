import cv2
import mediapipe as mp
import numpy as np

# Inicjalizacja MediaPipe (zgodnie z wersją 0.10.9)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """Oblicza kąt w stawie na podstawie współrzędnych 2D."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


cap = cv2.VideoCapture(0)

# Ustawienia Pose - optymalizacja pod analizę rzutu
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Konwersja kolorów dla MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Pobranie punktów prawej strony (najlepiej widocznej pod kątem 45 stopni z prawej)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Obliczenie kąta łokcia
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Logika Cyber Trenera: weryfikacja fazy przygotowania (optymalnie ok. 90 stopni)
            color = (0, 255, 0) if 80 <= elbow_angle <= 100 else (0, 0, 255)

            # Wyświetlanie danych na obrazie
            h, w, _ = image.shape
            pos = tuple(np.multiply(elbow, [w, h]).astype(int))
            cv2.putText(image, f"{int(elbow_angle)} deg", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        except Exception as e:
            pass

        # Rysowanie szkieletu
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Cyber Trener - Analiza rzutu 45 stopni', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()