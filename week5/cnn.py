import cv2 as cv
import numpy as np

img=cv.imread('images/soccer.jpg')
img=cv.resize(img, dsize=(0,0),fx=0.4,fy=0.4)   #0.2 축소 가능
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY) # color:회색조
cv.putText(gray,'soccer',(10,20),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)  # 255,255,255":흰색
cv.imshow('Original', gray)

smooth=np.hstack((cv.GaussianBlur(gray, (5,5), 0.0), cv.GaussianBlur(gray, (9,9), 0.0), 
                  cv.GaussianBlur(gray, (15,15), 0.0))) # 가우시안 필터 적용, 클수록 더 흐려지게 됨
cv.imshow('Smooth',smooth)

femboss=np.array([[-1.0, 0.0, 0.0], # 엠보싱 필터 정의
                  [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])

gray16 = np.int16(gray)

emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255)) # 0~255로 범위 제한->자연스로운 이미지
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss) + 128) # ->오버플로우, 언더플로우 전차 발생
emboss_worse = cv.filter2D(gray, -1, femboss) #-> 음수부분은 검정 256보다 크면 흰색

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()