from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
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
        parent=styles['Heading1'],
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
        spaceAfter=8
    )
    
    story = []
    
    story.append(Paragraph("照片修复与分析报告", title_style))
    story.append(Spacer(1, 20))
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"生成时间: {current_time}", normal_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("一、图像修复对比", heading_style))
    story.append(Spacer(1, 10))
    
    image_data = []
    max_img_width = 3 * inch
    max_img_height = 4 * inch
    
    original_size = resize_image(original_image_path, max_img_width, max_img_height)
    restored_size = resize_image(restored_image_path, max_img_width, max_img_height)
    
    if original_size and os.path.exists(original_image_path):
        original_img = Image(original_image_path, width=original_size[0], height=original_size[1])
        original_caption = Paragraph("修复前", normal_style)
        image_data.append([original_img, original_caption])
    else:
        image_data.append([Paragraph("原图无法加载", normal_style), ""])
    
    if restored_size and os.path.exists(restored_image_path):
        restored_img = Image(restored_image_path, width=restored_size[0], height=restored_size[1])
        restored_caption = Paragraph("修复后", normal_style)
        image_data.append([restored_img, restored_caption])
    else:
        image_data.append([Paragraph("修复图无法加载", normal_style), ""])
    
    image_table = Table(image_data, colWidths=[4 * inch, 1 * inch])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(image_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("二、OCR文字识别结果", heading_style))
    story.append(Spacer(1, 10))
    
    if recognized_text and recognized_text.strip():
        text_paragraph = Paragraph(
            f"<code>{recognized_text.replace('\\n', '<br/>')}</code>", 
            normal_style
        )
        story.append(text_paragraph)
    else:
        story.append(Paragraph("无文字内容", normal_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("三、情感分析结果", heading_style))
    story.append(Spacer(1, 10))
    
    if sentiment_result.get("has_text", False):
        polarity = sentiment_result["polarity"]
        subjectivity = sentiment_result["subjectivity"]
        sentiment = sentiment_result["sentiment"]
        
        sentiment_data = [
            ["分析项", "结果"],
            ["情感倾向", sentiment],
            ["极性分数", f"{polarity:.4f}"],
            ["主观性分数", f"{subjectivity:.4f}"],
        ]
        
        sentiment_table = Table(sentiment_data, colWidths=[2 * inch, 3 * inch])
        sentiment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
        ]))
        story.append(sentiment_table)
        
        story.append(Spacer(1, 10))
        story.append(Paragraph("极性分数说明: -1 (最负面) 到 +1 (最正面), 0 表示中性", normal_style))
        story.append(Paragraph("主观性分数说明: 0 (非常客观) 到 1 (非常主观)", normal_style))
    else:
        story.append(Paragraph("未检测到文字，无法进行情感分析", normal_style))
    story.append(Spacer(1, 20))
    
    if sentiment_result.get("has_text", False) and wordcloud_path and os.path.exists(wordcloud_path):
        story.append(Paragraph("四、词云图", heading_style))
        story.append(Spacer(1, 10))
        
        wc_size = resize_image(wordcloud_path, 6 * inch, 4 * inch)
        if wc_size:
            wc_img = Image(wordcloud_path, width=wc_size[0], height=wc_size[1])
            story.append(wc_img)
        else:
            story.append(Paragraph("词云图加载失败", normal_style))
    
    doc.build(story)
    print(f"PDF报告已生成: {output_path}")
    return True

if __name__ == "__main__":
    generate_pdf_report(
        original_image_path="1.jpg",
        restored_image_path="restored.jpg",
        recognized_text="This is a test text for the report.",
        sentiment_result={
            "polarity": 0.5,
            "subjectivity": 0.8,
            "sentiment": "正面",
            "has_text": True
        },
        wordcloud_path="wordcloud.png",
        output_path="test_report.pdf"
    )
