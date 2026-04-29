import cv2
import numpy as np

def detect_line_scratches(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    lines_mask = np.zeros_like(black_mask)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    for color_mask in [black_mask, red_mask]:
        if np.sum(color_mask) == 0:
            continue
            
        temp_mask = color_mask.copy()
        
        horizontal_lines = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        detected_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            
            if aspect_ratio > 3:
                cv2.drawContours(lines_mask, [contour], 0, 255, -1)
    
    kernel = np.ones((2, 2), np.uint8)
    final_mask = cv2.dilate(lines_mask, kernel, iterations=1)
    
    return img, final_mask, black_mask, red_mask

def inpaint_image_precise(img, mask):
    if np.sum(mask) == 0:
        return img.copy()
    
    restored = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return restored

def restore_image(image_path, output_path):
    img, mask, black_mask, red_mask = detect_line_scratches(image_path)
    
    if np.sum(mask) == 0:
        print("警告: 未检测到明显的划痕，将使用原图")
        restored = img.copy()
    else:
        restored = inpaint_image_precise(img, mask)
    
    cv2.imwrite(output_path, restored)
    
    mask_vis = np.zeros_like(img)
    mask_vis[:, :, 2] = mask
    cv2.imwrite("debug_mask.jpg", mask_vis)
    
    return img, restored, mask

if __name__ == "__main__":
    original, restored, mask = restore_image("1.jpg", "restored.jpg")
    print("图像修复完成，已保存为 restored.jpg")
    print(f"检测到的划痕区域像素数: {np.sum(mask > 0)}")
