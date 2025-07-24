# BLIP-2 + EasyOCR + EXAONE 모델을 사용하여 이미지 분석 스크립트

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import easyocr
import torch

# 🔷 분석할 이미지 파일 경로
image_path = "C:/test/dd.png"

# 1️⃣ BLIP-2: 이미지 → 간단한 캡션
def image_to_caption(image_path):
    print("[INFO] BLIP-2: 이미지 캡션 생성 중...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# 2️⃣ OCR: 이미지 → 손글씨 텍스트
def image_to_text(image_path):
    print("[INFO] EasyOCR: 손글씨 텍스트 추출 중...")
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    result = reader.readtext(image_path, detail=0)
    return ' '.join(result)

# 3️⃣ EXAONE 모델 기반 분석
def exaone_detailed_analysis(caption, ocr_text):
    print("[INFO] EXAONE: 상세 묘사 및 분석 요청 중...")

    # 프롬프트 구성
    prompt = f"""
다음은 어린이의 그림일기입니다.

[BLIP-2 캡션]
{caption}

[OCR 텍스트]
{ocr_text}

위 두 가지 정보를 바탕으로 이미지의 상황을 사람이 보는 것처럼 매우 자세히 묘사해 주세요.
예: 해가 웃고 있다, 자동차가 있다, 자동차 안에 사람이 3명이 있다, 엄마, 아빠, 어린아이가 있다, 꽃이 있다.

그 후, 해당 상황에서 느껴지는 감정, 원인, 관계를 다음 형식으로 분석해 주세요:

[상황 묘사]
…

[감정]
…

[원인]
…

[관계]
…
"""

    # EXAONE 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 첫 번째 줄에 prompt까지 같이 포함될 수 있으므로 분리
    if "[상황 묘사]" in response:
        response = response.split("[상황 묘사]")[1]
        response = "[상황 묘사]" + response.strip()

    return response

# ▶️ 메인 실행
if __name__ == "__main__":
    # 1. 이미지 → 캡션
    caption = image_to_caption(image_path)
    print(f"\n[BLIP-2 캡션]\n{caption}")

    # 2. OCR 텍스트 추출
    ocr_text = image_to_text(image_path)
    print(f"\n[OCR 텍스트]\n{ocr_text}")

    # 3. EXAONE 분석
    result = exaone_detailed_analysis(caption, ocr_text)
    print("\n[EXAONE 분석 결과]")
    print(result)
