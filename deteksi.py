import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load model hasil training
model = YOLO("best.pt")  # Ganti dengan path model Anda

# Buka kamera USB (0 = kamera default, 1 = kamera eksternal)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Jalankan deteksi dengan YOLOv8
    results = model(frame)

    # Loop melalui hasil deteksi
    for result in results:
        for box in result.boxes:
            # Ambil koordinat bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  

            # Kelas deteksi
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Confidence Score (akurasi)
            confidence = float(box.conf[0]) * 100

            # Hitung titik tengah bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Tampilkan koordinat (x, y) di bounding box
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({center_x}, {center_y})", (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Print hasil deteksi di terminal
            print(f"Deteksi: {label} | Confidence: {confidence:.2f}% | Koordinat: ({center_x}, {center_y})")

    # Tampilkan hasil deteksi di jendela
    cv2.imshow("YOLOv8 - Jetson Nano", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
