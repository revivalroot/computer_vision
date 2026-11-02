import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import winsound

model = tf.keras.models.load_model('dmlp_trained.h5')

def reset():
    global img

    img = np.ones((200, 520, 3), dtype=np.uint8) * 255 # 창의 흰 배경
    for i in range(5):
        cv.rectangle(img, (10 + i * 100, 50), (10 + (i + 1) * 100, 150), (0, 0, 255)) # 빨간 사각형 5개
    cv.putText(img, 'e:erase s:show r:recognition q:quit', (10, 40), # 안내 텍스트
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)

def grab_numerals():
    numerals = []
    for i in range(5):
        roi = img[51:149, 11 + i * 100:9 + (i + 1) * 100, 0] # 각 칸의 ROI 자르기
        roi = 255 - cv.resize(roi, (28, 28), interpolation=cv.INTER_CUBIC) # 색상 반전, 28x28사이즈로 변경
        numerals.append(roi)
    numerals = np.array(numerals)
    return numerals

def show(): # s를 눌러 잘라낸 숫자를 확인하기
    numerals = grab_numerals()
    plt.figure(figsize=(25, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(numerals[i], cmap='gray')
        plt.xticks([]); plt.yticks([])
    plt.show()

def recognition():  # r키를 눌러 숫자 인식하기
    numerals = grab_numerals()
    numerals = numerals.reshape(5, 784)
    numerals = numerals.astype(np.float32) / 255.0
    res = model.predict(numerals)             # 신경망 모델로 예측
    class_id = np.argmax(res, axis=1)
    for i in range(5):
        cv.putText(img, str(class_id[i]), (50 + i * 100, 180), # 인식결과를 각 칸 아래에 표시
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    winsound.Beep(1000, 500)

BrushSiz = 4
LColor = (0, 0, 0)

def writing(event, x, y, flags, param): # 마우스로 그림 그리기 // 마우스 콜백 함수
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)

reset()
cv.namedWindow('Writing')
cv.setMouseCallback('Writing', writing)

while(True):
    cv.imshow('Writing', img)
    key = cv.waitKey(1)
    if key == ord('e'):
        reset()
    elif key == ord('s'):
        show()
    elif key == ord('r'):
        recognition()
    elif key == ord('q'):
        break

cv.destroyAllWindows()