from cv2 import cv2
import numpy as np

green = np.array([0, 255, 0], dtype=np.uint8)
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

green_mask = np.where(np.all(1, axis=-1,
                             keepdims=True), green, frame1)
clock = 50
while True:

    if 50 <= clock < 100:

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame1 = cv2.addWeighted(frame1, 0.8, green_mask, 0.3, 1)

        cnt = 0
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                cnt += 1
                continue
            cv2.drawContours(frame1, contours, cnt, (0, 0, 255), -1)
            cnt += 1
        text = 'RED LIGHT ' + str(round(5 - (clock - 50) / 10, 1))
        cv2.putText(frame1, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        frame1 = cv2.addWeighted(frame1, 0.8, green_mask, 0.3, 1)
        text = 'GREEN LIGHT ' + str(str(round(5 - (clock / 10), 1)))
        cv2.putText(frame1, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("camera", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()
    clock += 1
    clock %= 100

    key = cv2.waitKey(20) & 0xff
    if key == 27:  # Esc
        break

cv2.destroyAllWindows()
cap.release()
