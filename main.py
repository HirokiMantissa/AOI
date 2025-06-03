import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from cnn import CardClassifier 
import os

image = cv2.imread('test4.png')

def extract_cards(img):
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
    card_images = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            card = img[y:y+h, x:x+w]
            card_images.append(card)
            filtered_contours.append(cnt)
    #====  end  ====
    
    cv2.drawContours(img, filtered_contours, -1, (255,0,255), 2)
    cv2.imshow("all contours", img)
    cv2.waitKey(0)
    
    return card_images


if image is None:
    print("讀取圖片失敗！請確認檔案路徑是否正確")
else:
    width = 1080
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))

    card_images = extract_cards(resized_image.copy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CardClassifier(num_classes=13).to(device)
model.load_state_dict(torch.load("card_classifier.pth", map_location=device))
model.eval()

# 建立轉換流程：BGR 陣列轉 tensor
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 推論每張卡牌
for idx, card_bgr in enumerate(card_images):
    card_rgb = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB)
    cv2.imshow("target", card_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pil_img = Image.fromarray(card_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)  # shape: [1, 1, 28, 28]

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        print(f"卡牌 {idx+1} 預測結果：{pred}")