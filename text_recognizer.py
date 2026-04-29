import pytesseract
import cv2
import os
import numpy as np

TESSERACT_PATH = r"D:\Ocr\tesseract.exe"

def setup_tesseract():
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        raise FileNotFoundError(f"未找到Tesseract可执行文件: {TESSERACT_PATH}")

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def denoise_image(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

def adaptive_threshold(gray):
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def otsu_threshold(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def preprocess_for_ocr(image_path, method="all"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    preprocessed_images = {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed_images["gray"] = gray
    
    enhanced = enhance_contrast(img)
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    preprocessed_images["enhanced_gray"] = gray_enhanced
    
    sharpened = sharpen_image(img)
    gray_sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    preprocessed_images["sharpened_gray"] = gray_sharpened
    
    preprocessed_images["adaptive_thresh"] = adaptive_threshold(gray)
    preprocessed_images["adaptive_thresh_enhanced"] = adaptive_threshold(gray_enhanced)
    preprocessed_images["otsu_thresh"] = otsu_threshold(gray)
    preprocessed_images["otsu_thresh_enhanced"] = otsu_threshold(gray_enhanced)
    
    denoised = denoise_image(img)
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    preprocessed_images["denoised_gray"] = gray_denoised
    
    inverted = cv2.bitwise_not(gray)
    preprocessed_images["inverted"] = inverted
    
    return preprocessed_images

def recognize_with_config(image, configs, lang='eng'):
    results = []
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config, lang=lang)
            text = text.strip()
            if text:
                results.append({
                    'text': text,
                    'config': config,
                    'length': len(text),
                    'word_count': len(text.split())
                })
        except Exception as e:
            continue
    
    return results

def select_best_result(results):
    if not results:
        return ""
    
    sorted_results = sorted(results, key=lambda x: (x['word_count'], x['length']), reverse=True)
    
    return sorted_results[0]['text']

def recognize_text(image_path):
    setup_tesseract()
    
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 11',
        r'--oem 3 --psm 12',
        r'--oem 3 --psm 3',
        r'--oem 3 --psm 4',
        r'--oem 1 --psm 6',
    ]
    
    all_results = []
    
    try:
        preprocessed = preprocess_for_ocr(image_path)
        
        for name, img in preprocessed.items():
            results = recognize_with_config(img, configs)
            for r in results:
                r['preprocess'] = name
            all_results.extend(results)
            
    except Exception as e:
        print(f"预处理识别失败: {e}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = recognize_with_config(gray, configs)
        for r in results:
            r['preprocess'] = 'fallback_gray'
        all_results.extend(results)
    
    best_text = select_best_result(all_results)
    
    if best_text:
        print(f"最佳识别结果来自: {next((r['preprocess'] for r in all_results if r['text'] == best_text), 'unknown')}")
    
    return best_text if best_text else ""

if __name__ == "__main__":
    text = recognize_text("1.jpg")
    if text:
        print("识别到的文字:")
        print("-" * 40)
        print(text)
        print("-" * 40)
    else:
        print("无文字内容")
