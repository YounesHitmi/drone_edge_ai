import cv2

# 0 is the default webcam. If it is a black screen, test 1 or 2.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error : impossible to access camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('My Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()