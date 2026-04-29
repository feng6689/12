import pytesseract
import cv2
import os

TESSERACT_PATH = r"D:\Ocr\tesseract.exe"

def setup_tesseract():
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        raise FileNotFoundError(f"未找到Tesseract可执行文件: {TESSERACT_PATH}")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((1, 1), np.uint8)
    img_dilated = cv2.dilate(thresh, kernel, iterations=1)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)
    
    return img_eroded

import numpy as np

def recognize_text(image_path):
    setup_tesseract()
    
    try:
        preprocessed = preprocess_image(image_path)
    except Exception:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        preprocessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed, config=custom_config, lang='eng')
    
    text = text.strip()
    
    if not text:
        return ""
    
    return text

if __name__ == "__main__":
    text = recognize_text("restored.jpg")
    if text:
        print("识别到的文字:")
        print(text)
    else:
        print("无文字内容")
