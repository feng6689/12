from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime

def resize_image(image_path, max_width, max_height):
    from PIL import Image as PILImage
    
    if not os.path.exists(image_path):
        return None
    
    try:
        img = PILImage.open(image_path)
        original_width, original_height = img.size
        
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        ratio = min(width_ratio, height_ratio)
        
        new_width = original_width * ratio
        new_height = original_height * ratio
        
        return new_width, new_height
    except Exception:
        return None

def generate_pdf_report(
    original_image_path,
    restored_image_path,
    recognized_text,
    sentiment_result,
    wordcloud_path="wordcloud.png",
    output_path="report.pdf"
):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=20,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=11,
        spaceAfter=8,
        leading=14
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['BodyText'],
        fontName='Courier',
        fontSize=10,
        spaceAfter=8,
        leading=13,
        leftIndent=20
    )
    
    story = []
    
    story.append(Paragraph("Photo Restoration and Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated at: {current_time}", normal_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1. Image Restoration Comparison", heading_style))
    story.append(Spacer(1, 10))
    
    image_data = []
    max_img_width = 3 * inch
    max_img_height = 4 * inch
    
    original_size = resize_image(original_image_path, max_img_width, max_img_height)
    restored_size = resize_image(restored_image_path, max_img_width, max_img_height)
    
    if original_size and os.path.exists(original_image_path):
        original_img = Image(original_image_path, width=original_size[0], height=original_size[1])
        original_caption = Paragraph("Before Restoration", normal_style)
        image_data.append([original_img, original_caption])
    else:
        image_data.append([Paragraph("Original image not available", normal_style), ""])
    
    if restored_size and os.path.exists(restored_image_path):
        restored_img = Image(restored_image_path, width=restored_size[0], height=restored_size[1])
        restored_caption = Paragraph("After Restoration", normal_style)
        image_data.append([restored_img, restored_caption])
    else:
        image_data.append([Paragraph("Restored image not available", normal_style), ""])
    
    image_table = Table(image_data, colWidths=[4 * inch, 1.5 * inch])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("2. OCR Text Recognition Results", heading_style))
    story.append(Spacer(1, 10))
    
    if recognized_text and recognized_text.strip():
        lines = recognized_text.split('\n')
        for line in lines:
            if line.strip():
                safe_line = line.replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe_line, code_style))
            else:
                story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No text content", normal_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("3. Sentiment Analysis Results", heading_style))
    story.append(Spacer(1, 10))
    
    if sentiment_result.get("has_text", False):
        polarity = sentiment_result["polarity"]
        subjectivity = sentiment_result["subjectivity"]
        sentiment = sentiment_result["sentiment"]
        
        if sentiment == "正面" or sentiment == "Positive":
            sentiment_display = "Positive"
        elif sentiment == "负面" or sentiment == "Negative":
            sentiment_display = "Negative"
        else:
            sentiment_display = "Neutral"
        
        sentiment_data = [
            ["Item", "Result"],
            ["Sentiment", sentiment_display],
            ["Polarity Score", f"{polarity:.4f}"],
            ["Subjectivity Score", f"{subjectivity:.4f}"],
        ]
        
        sentiment_table = Table(sentiment_data, colWidths=[2 * inch, 3 * inch])
        sentiment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E7E6E6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ]))
        story.append(sentiment_table)
        
        story.append(Spacer(1, 10))
        story.append(Paragraph("Polarity: -1 (Most Negative) to +1 (Most Positive), 0 = Neutral", normal_style))
        story.append(Paragraph("Subjectivity: 0 (Very Objective) to 1 (Very Subjective)", normal_style))
    else:
        story.append(Paragraph("No text detected, sentiment analysis skipped", normal_style))
    story.append(Spacer(1, 20))
    
    if sentiment_result.get("has_text", False) and wordcloud_path and os.path.exists(wordcloud_path):
        story.append(Paragraph("4. Word Cloud", heading_style))
        story.append(Spacer(1, 10))
        
        wc_size = resize_image(wordcloud_path, 6 * inch, 4 * inch)
        if wc_size:
            wc_img = Image(wordcloud_path, width=wc_size[0], height=wc_size[1])
            story.append(wc_img)
        else:
            story.append(Paragraph("Word cloud image failed to load", normal_style))
    
    doc.build(story)
    print(f"PDF report generated: {output_path}")
    return True

if __name__ == "__main__":
    generate_pdf_report(
        original_image_path="1.jpg",
        restored_image_path="restored.jpg",
        recognized_text="This is a test text for the report. Hello World.",
        sentiment_result={
            "polarity": 0.5,
            "subjectivity": 0.8,
            "sentiment": "Positive",
            "has_text": True
        },
        wordcloud_path="wordcloud.png",
        output_path="test_report.pdf"
    )
