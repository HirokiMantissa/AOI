import cv2
import numpy as np

image = cv2.imread('test4.png')

def draw_all_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #==== filter ====
    min_area = 10000
    max_area = 30000
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            filtered_contours.append(cnt)
    #====  end  ====
    
    cv2.drawContours(img, filtered_contours, -1, (255,0,255), 2)
    
    return img


if image is None:
    print("讀取圖片失敗！請確認檔案路徑是否正確")
else:
    width = 1080
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))

    frame_with_contours = draw_all_contours(resized_image.copy())
    cv2.imshow("all contours", frame_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
