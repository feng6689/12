import cv2
import numpy as np

def detect_scratches_simple(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    lower_red1 = np.array([0, 80, 70])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([140, 80, 70])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area > 5:
            cv2.drawContours(mask, [contour], 0, 255, -1)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    combined_black = cv2.bitwise_or(black_mask, binary)
    
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    h_lines = cv2.morphologyEx(combined_black, cv2.MORPH_OPEN, kernel_h, iterations=1)
    v_lines = cv2.morphologyEx(combined_black, cv2.MORPH_OPEN, kernel_v, iterations=1)
    
    line_mask = cv2.bitwise_or(h_lines, v_lines)
    
    line_contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in line_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area > 10:
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            
            if aspect_ratio > 1.5:
                cv2.drawContours(mask, [contour], 0, 255, -1)
    
    black_contours, _ = cv2.findContours(combined_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in black_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area > 5:
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            
            if aspect_ratio > 2:
                cv2.drawContours(mask, [contour], 0, 255, -1)
    
    kernel_dilate = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    return img, mask

def inpaint_image(img, mask):
    if np.sum(mask) == 0:
        return img.copy()
    
    restored_telea = cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    
    return restored_telea

def restore_image(image_path, output_path):
    img, mask = detect_scratches_simple(image_path)
    
    mask_pixels = np.sum(mask > 0)
    print(f"检测到的划痕区域: {mask_pixels} 像素")
    
    if mask_pixels == 0:
        print("警告: 未检测到明显的划痕，将使用原图")
        restored = img.copy()
    else:
        restored = inpaint_image(img, mask)
    
    cv2.imwrite(output_path, restored)
    
    mask_vis = np.zeros_like(img)
    mask_vis[:, :, 2] = mask
    cv2.imwrite("debug_mask.jpg", mask_vis)
    
    return img, restored, mask

if __name__ == "__main__":
    original, restored, mask = restore_image("1.jpg", "restored.jpg")
    print("图像修复完成，已保存为 restored.jpg")
    print(f"检测到的划痕区域像素数: {np.sum(mask > 0)}")
