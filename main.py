import numpy as np
import cv2
import os

cap = cv2.VideoCapture('data.mp4')

save_id = 0
os.makedirs("DATA", exist_ok=True)
##############################################################################################
def preprocess(img):
    img = cv2.resize(img, (190, 270))  # 縮放到標準撲克牌大小
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 擷取左上角（數字+花色）區域
    corner = img[2:95, 11:45]  # 可微調
    corner = cv2.resize(corner, (70, 100))  # 和模板一致
    return corner

def match_template(img, template_dir):
    best_score = None
    best_match = "?"
    img = preprocess(img)
    for name in os.listdir(template_dir):
        template = cv2.imread(os.path.join(template_dir, name), cv2.IMREAD_GRAYSCALE)
        
        if template is None or template.shape[0] == 0:
            continue
        template = preprocess(template)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        
        if best_score is None or score > best_score:
            best_score = score
            best_match = os.path.splitext(name)[0]
    return best_match




###########################################################################################################
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.medianBlur(img_gray,1)
    img_binary = cv2.adaptiveThreshold(
        img_blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        13,2
    )
    
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    corner_list = []
    rank_list = []
    suit_list = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 4000:  # 太小的雜訊忽略
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (x, y), (w, h), angle = rect
        ratio = max(w, h) / min(w, h)
        if ratio < 1.2 or ratio > 1.7:
            continue
        if w > h:
            angle += 90
            w, h = h, w

        # 對牌進行透視變換（warp）
        M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        card = cv2.getRectSubPix(rotated, (int(w), int(h)), (int(x), int(y)))
        

        card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        card_gray= cv2.resize(card_gray, (190, 270))
        
        corner_list.append(card_gray)
        card_save_path = f"DATA/card_{save_id}.jpg"
        #cv2.imwrite(card_save_path, card_gray)
        save_id += 1
        # 擷取左上角（花色+數字區）
        corner = card_gray[2:95, 11:45]
        
        corner = cv2.resize(corner, (160, 400))

        # 模板比對
        card = match_template(card_gray, 'cards')

        # 標籤
        cv2.putText(frame, f"{card}", (box[0][0], box[0][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # 使用 drawContours 畫出四邊形框
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        # 橫向排列（也可用 vstack）
    if len(corner_list) > 0:
        corners_display = np.hstack(corner_list)
        cv2.imshow("All Corners", corners_display)


    cv2.imshow('output',frame) # 顯示原圖＋標註框
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()