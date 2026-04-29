import os
import sys

def count_valid_words(text):
    if not text or not text.strip():
        return 0
    words = text.split()
    valid_words = [w for w in words if len(w) >= 2 and any(c.isalpha() for c in w)]
    return len(valid_words)

def main():
    print("=" * 50)
    print("照片修复与分析流程开始")
    print("=" * 50)
    
    input_image = "1.jpg"
    restored_image = "restored.jpg"
    wordcloud_image = "wordcloud.png"
    pdf_report = "report.pdf"
    
    if not os.path.exists(input_image):
        print(f"错误: 未找到输入图片 {input_image}")
        sys.exit(1)
    
    try:
        print("\n[步骤 1/5] 图像修复中...")
        from image_restorer import restore_image
        
        original, restored, mask = restore_image(input_image, restored_image)
        print(f"图像修复完成，已保存为: {restored_image}")
        print(f"检测到的划痕区域: {mask.sum() // 255} 像素")
        
    except Exception as e:
        print(f"图像修复失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    recognized_text = ""
    text_source = ""
    
    try:
        print("\n[步骤 2/5] OCR文字识别中...")
        from text_recognizer import recognize_text
        
        print("\n正在识别原图...")
        text_original = recognize_text(input_image)
        count_original = count_valid_words(text_original)
        
        print("\n正在识别修复后图片...")
        text_restored = recognize_text(restored_image)
        count_restored = count_valid_words(text_restored)
        
        print("\n" + "-" * 40)
        print(f"原图识别有效单词数: {count_original}")
        print(f"修复图识别有效单词数: {count_restored}")
        
        if count_original >= count_restored and text_original:
            recognized_text = text_original
            text_source = "原图"
        elif text_restored:
            recognized_text = text_restored
            text_source = "修复后图片"
        else:
            recognized_text = ""
            text_source = "无"
        
        print(f"选择使用: {text_source}")
        print("-" * 40)
        
        if recognized_text:
            print("\n识别到的文字:")
            print("-" * 40)
            print(recognized_text)
            print("-" * 40)
        else:
            print("识别结果: 无文字内容")
            
    except Exception as e:
        print(f"OCR识别失败: {e}")
        import traceback
        traceback.print_exc()
        recognized_text = ""
    
    sentiment_result = {
        "polarity": 0.0,
        "subjectivity": 0.0,
        "sentiment": "中性",
        "has_text": False
    }
    
    has_wordcloud = False
    if recognized_text and recognized_text.strip():
        try:
            print("\n[步骤 3/5] 情感分析中...")
            from sentiment_analyzer import analyze_sentiment
            
            sentiment_result = analyze_sentiment(recognized_text)
            print(f"情感倾向: {sentiment_result['sentiment']}")
            print(f"极性分数: {sentiment_result['polarity']:.4f}")
            print(f"主观性分数: {sentiment_result['subjectivity']:.4f}")
            sentiment_result["has_text"] = True
            
        except Exception as e:
            print(f"情感分析失败: {e}")
            import traceback
            traceback.print_exc()
            sentiment_result["has_text"] = False
        
        try:
            print("\n[步骤 4/5] 生成词云图中...")
            from sentiment_analyzer import generate_wordcloud
            
            has_wordcloud = generate_wordcloud(recognized_text, wordcloud_image)
            if has_wordcloud:
                print(f"词云图已生成: {wordcloud_image}")
            else:
                print("词云图生成失败或无有效词汇")
                
        except Exception as e:
            print(f"词云生成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[步骤 3/5] 无文字内容，跳过情感分析")
        print("[步骤 4/5] 无文字内容，跳过词云生成")
    
    try:
        print("\n[步骤 5/5] 生成PDF报告中...")
        from report_generator import generate_pdf_report
        
        generate_pdf_report(
            original_image_path=input_image,
            restored_image_path=restored_image,
            recognized_text=recognized_text,
            sentiment_result=sentiment_result,
            wordcloud_path=wordcloud_image if has_wordcloud else None,
            output_path=pdf_report
        )
        
    except Exception as e:
        print(f"PDF报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("处理完成!")
    print("=" * 50)
    print("\n生成的文件:")
    if os.path.exists(restored_image):
        print(f"  - 修复后的图像: {restored_image}")
    if has_wordcloud and os.path.exists(wordcloud_image):
        print(f"  - 词云图: {wordcloud_image}")
    if os.path.exists(pdf_report):
        print(f"  - PDF报告: {pdf_report}")
    if os.path.exists("debug_mask.jpg"):
        print(f"  - 调试用mask图像: debug_mask.jpg")
    
    print("\n流程执行完毕!")

if __name__ == "__main__":
    main()
