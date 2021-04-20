# opencv3.1 ,python3.5.2
import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False);

cv2.bgsegm.createBackgroundSubtractorMOG
fgbg = cv2.createBackgroundSubtractorMOG2()

background_capture = cv2.VideoCapture('matrix2.mp4')
cap = cv2.VideoCapture(0)

while True:

    ret, background = background_capture.read()
    # print(background.shape)
    background = cv2.resize(background, (640, 480), interpolation=cv2.INTER_AREA)
    #background = cv2.resize(background, (1920,1080 ), interpolation=cv2.INTER_AREA)
    ret, frame = cap.read()
    # print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = fgbg.apply(frame, 1)
    bitwise = cv2.bitwise_and(background, background, mask=mask)


    #cv2.cvtColor(src, src_gray, COLOR_BGR2GRAY)
    detected_edges =cv2.blur(gray, (5, 5))
    edge1 = cv2.Canny(detected_edges, 100, 200, L2gradient = True)
    mask2 = edge1 != 0
  # dst = src * (mask[:, :, None].astype(src.dtype))
    dst = frame * (mask2[:, :, None].astype(frame.dtype))
  #gaussian = cv2.GaussianBlur(bitwise, (5, 5), 10)
    gaussian = cv2.GaussianBlur(bitwise, (5, 5), 10)
    add = cv2.add(background, gaussian)
   # add = cv2.add(background, dst)
    add = cv2.resize(add, (1366, 768), interpolation=cv2.INTER_AREA)

    cv2.imshow('background', background)
    cv2.imshow('frame', frame)
    cv2.imshow('canny2', dst)
    cv2.imshow('bitwise', bitwise)
    cv2.imshow('gaussian', gaussian)
    cv2.imshow('canny', edge1)
  #  cv2('blur', detected_edges)
    cv2.imshow('add', add)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# When everything is done, release the capture
cap.video_capture.release()
cv2.destroyAllWindows()
