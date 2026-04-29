import cv2
import numpy as np

def detect_scratches(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    lower_red1 = np.array([0, 100, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 70])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    combined_mask = cv2.bitwise_or(black_mask, red_mask)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    return img, dilated_mask

def inpaint_image(img, mask):
    restored = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return restored

def restore_image(image_path, output_path):
    img, mask = detect_scratches(image_path)
    restored = inpaint_image(img, mask)
    cv2.imwrite(output_path, restored)
    return img, restored, mask

if __name__ == "__main__":
    original, restored, mask = restore_image("1.jpg", "restored.jpg")
    print("图像修复完成，已保存为 restored.jpg")
