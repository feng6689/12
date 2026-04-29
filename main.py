import os
import sys

def count_valid_words(text):
    if not text or not text.strip():
        return 0
    words = text.split()
    valid_words = [w for w in words if len(w) >= 2 and w.isalpha()]
    return len(valid_words)

def main():
    print("=" * 60)
    print("Photo Restoration and Analysis Pipeline")
    print("=" * 60)
    
    input_image = "1.jpg"
    restored_image = "restored.jpg"
    wordcloud_image = "wordcloud.png"
    pdf_report = "report.pdf"
    
    if not os.path.exists(input_image):
        print(f"Error: Input image not found: {input_image}")
        sys.exit(1)
    
    try:
        print("\n[Step 1/5] Image Restoration...")
        from image_restorer import restore_image
        
        original, restored, mask = restore_image(input_image, restored_image)
        print(f"Image restored, saved as: {restored_image}")
        print(f"Detected scratch pixels: {mask.sum() // 255}")
        
    except Exception as e:
        print(f"Image restoration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    recognized_text = ""
    text_source = ""
    best_valid_count = 0
    
    try:
        print("\n[Step 2/5] OCR Text Recognition...")
        from text_recognizer import recognize_text
        
        print("\n" + "-" * 60)
        print("Recognizing text from ORIGINAL image...")
        print("-" * 60)
        text_original = recognize_text(input_image)
        count_original = count_valid_words(text_original)
        
        print("\n" + "-" * 60)
        print("Recognizing text from RESTORED image...")
        print("-" * 60)
        text_restored = recognize_text(restored_image)
        count_restored = count_valid_words(text_restored)
        
        print("\n" + "=" * 60)
        print(f"Original image valid words: {count_original}")
        print(f"Restored image valid words: {count_restored}")
        
        if count_original >= count_restored and text_original:
            recognized_text = text_original
            text_source = "Original Image"
            best_valid_count = count_original
        elif text_restored:
            recognized_text = text_restored
            text_source = "Restored Image"
            best_valid_count = count_restored
        else:
            recognized_text = ""
            text_source = "None"
        
        print(f"Selected source: {text_source}")
        print("=" * 60)
        
        if recognized_text and best_valid_count > 0:
            print("\nFinal recognized text:")
            print("-" * 40)
            print(recognized_text)
            print("-" * 40)
        else:
            print("\nNo valid text recognized")
            
    except Exception as e:
        print(f"OCR recognition failed: {e}")
        import traceback
        traceback.print_exc()
        recognized_text = ""
    
    sentiment_result = {
        "polarity": 0.0,
        "subjectivity": 0.0,
        "sentiment": "Neutral",
        "has_text": False
    }
    
    has_wordcloud = False
    if recognized_text and best_valid_count > 0:
        try:
            print("\n[Step 3/5] Sentiment Analysis...")
            from sentiment_analyzer import analyze_sentiment
            
            sentiment_result = analyze_sentiment(recognized_text)
            
            sentiment_eng = sentiment_result.get("sentiment", "Neutral")
            if sentiment_eng == "正面":
                sentiment_eng = "Positive"
            elif sentiment_eng == "负面":
                sentiment_eng = "Negative"
            elif sentiment_eng == "中性":
                sentiment_eng = "Neutral"
            
            print(f"Sentiment: {sentiment_eng}")
            print(f"Polarity Score: {sentiment_result['polarity']:.4f}")
            print(f"Subjectivity Score: {sentiment_result['subjectivity']:.4f}")
            sentiment_result["has_text"] = True
            sentiment_result["sentiment"] = sentiment_eng
            
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            import traceback
            traceback.print_exc()
            sentiment_result["has_text"] = False
        
        try:
            print("\n[Step 4/5] Generating Word Cloud...")
            from sentiment_analyzer import generate_wordcloud
            
            has_wordcloud = generate_wordcloud(recognized_text, wordcloud_image)
            if has_wordcloud:
                print(f"Word cloud generated: {wordcloud_image}")
            else:
                print("Word cloud generation failed or no valid words")
                
        except Exception as e:
            print(f"Word cloud generation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Step 3/5] No valid text, skipping sentiment analysis")
        print("[Step 4/5] No valid text, skipping word cloud generation")
    
    try:
        print("\n[Step 5/5] Generating PDF Report...")
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
        print(f"PDF report generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    if os.path.exists(restored_image):
        print(f"  - Restored image: {restored_image}")
    if os.path.exists("debug_mask.jpg"):
        print(f"  - Debug mask image: debug_mask.jpg")
    if has_wordcloud and os.path.exists(wordcloud_image):
        print(f"  - Word cloud: {wordcloud_image}")
    if os.path.exists(pdf_report):
        print(f"  - PDF report: {pdf_report}")
    
    print("\n" + "=" * 60)
    print("Pipeline finished successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
