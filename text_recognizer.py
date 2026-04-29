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

def resize_image(image, scale):
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized

def to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def threshold_otsu(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def threshold_adaptive(gray):
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def invert(gray):
    return cv2.bitwise_not(gray)

def denoise_median(gray):
    return cv2.medianBlur(gray, 3)

def denoise_gaussian(gray):
    return cv2.GaussianBlur(gray, (3, 3), 0)

def sharpen(gray):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)

def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def create_preprocess_variants(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    variants = []
    
    gray = to_grayscale(img)
    
    for scale in [1, 2, 3, 4]:
        if scale == 1:
            scaled = gray
        else:
            scaled = resize_image(gray, scale)
        
        variants.append((f"scale_{scale}_gray", scaled))
        
        otsu = threshold_otsu(scaled)
        variants.append((f"scale_{scale}_otsu", otsu))
        
        adaptive = threshold_adaptive(scaled)
        variants.append((f"scale_{scale}_adaptive", adaptive))
        
        inverted = invert(scaled)
        variants.append((f"scale_{scale}_inverted", inverted))
        
        inverted_otsu = invert(otsu)
        variants.append((f"scale_{scale}_inverted_otsu", inverted_otsu))
        
        denoised = denoise_median(scaled)
        variants.append((f"scale_{scale}_denoised", denoised))
        
        denoised_otsu = threshold_otsu(denoised)
        variants.append((f"scale_{scale}_denoised_otsu", denoised_otsu))
        
        sharpened = sharpen(scaled)
        variants.append((f"scale_{scale}_sharpened", sharpened))
        
        sharpened_otsu = threshold_otsu(sharpened)
        variants.append((f"scale_{scale}_sharpened_otsu", sharpened_otsu))
        
        enhanced = enhance_contrast(scaled)
        variants.append((f"scale_{scale}_enhanced", enhanced))
        
        enhanced_otsu = threshold_otsu(enhanced)
        variants.append((f"scale_{scale}_enhanced_otsu", enhanced_otsu))
    
    return variants

def is_valid_word(word):
    if len(word) < 2:
        return False
    if not word.isalpha():
        return False
    return True

def count_valid_words(text):
    if not text or not text.strip():
        return 0
    words = text.split()
    valid_count = sum(1 for w in words if is_valid_word(w))
    return valid_count

def recognize_with_configs(image):
    configs = [
        r'--oem 3 --psm 3',
        r'--oem 3 --psm 4',
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 7',
        r'--oem 3 --psm 8',
        r'--oem 3 --psm 11',
        r'--oem 3 --psm 12',
        r'--oem 1 --psm 3',
        r'--oem 1 --psm 6',
        r'--oem 1 --psm 7',
        r'--oem 1 --psm 8',
    ]
    
    results = []
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config, lang='eng')
            text = text.strip()
            if text:
                valid_count = count_valid_words(text)
                total_words = len(text.split())
                results.append({
                    'text': text,
                    'config': config,
                    'valid_words': valid_count,
                    'total_words': total_words,
                    'length': len(text)
                })
        except Exception as e:
            continue
    
    return results

def select_best_result(all_results):
    if not all_results:
        return None, 0
    
    sorted_results = sorted(
        all_results,
        key=lambda x: (x['valid_words'], x['total_words'], x['length']),
        reverse=True
    )
    
    best = sorted_results[0]
    return best['text'], best['valid_words']

def recognize_text(image_path):
    setup_tesseract()
    
    print(f"\n开始OCR识别: {image_path}")
    
    try:
        variants = create_preprocess_variants(image_path)
    except Exception as e:
        print(f"预处理失败: {e}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        gray = to_grayscale(img)
        variants = [("fallback", gray)]
    
    all_results = []
    best_valid_count = 0
    best_text = ""
    best_variant = ""
    best_config = ""
    
    for name, img in variants:
        results = recognize_with_configs(img)
        
        for r in results:
            r['variant'] = name
            all_results.append(r)
            
            if r['valid_words'] > best_valid_count:
                best_valid_count = r['valid_words']
                best_text = r['text']
                best_variant = name
                best_config = r['config']
        
        if results:
            best_for_variant = max(results, key=lambda x: x['valid_words'])
            if best_for_variant['valid_words'] > 0:
                print(f"  {name}: 有效单词数={best_for_variant['valid_words']}")
    
    if best_valid_count == 0 and all_results:
        sorted_by_total = sorted(all_results, key=lambda x: (x['total_words'], x['length']), reverse=True)
        best = sorted_by_total[0]
        best_text = best['text']
        best_variant = best['variant']
        best_config = best['config']
        best_valid_count = best['valid_words']
    
    print(f"\n{'='*50}")
    print(f"最佳识别结果:")
    print(f"  预处理方式: {best_variant}")
    print(f"  Tesseract配置: {best_config}")
    print(f"  有效单词数: {best_valid_count}")
    print(f"{'='*50}")
    
    if best_text:
        print("\n识别到的文字:")
        print("-" * 40)
        print(best_text)
        print("-" * 40)
    else:
        print("\n未识别到有效文字")
    
    return best_text if best_text else ""

if __name__ == "__main__":
    text = recognize_text("1.jpg")
    if text:
        print("\n最终识别结果:")
        print(text)
    else:
        print("无文字内容")
