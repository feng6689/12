import os
import sys

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
        
    except Exception as e:
        print(f"图像修复失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    recognized_text = ""
    try:
        print("\n[步骤 2/5] OCR文字识别中...")
        from text_recognizer import recognize_text
        
        recognized_text = recognize_text(restored_image)
        
        if recognized_text:
            print("识别到的文字:")
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
            
        except Exception as e:
            print(f"情感分析失败: {e}")
            import traceback
            traceback.print_exc()
        
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
    
    print("\n流程执行完毕!")

if __name__ == "__main__":
    main()
