import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera device {i} is available")
    else:
        print(f"Camera device {i} is not available")
    cap.release()
