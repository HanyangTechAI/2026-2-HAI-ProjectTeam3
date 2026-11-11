import io
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import qrcode
import onnxruntime as ort

app = FastAPI(
    title="AI Timemachine Photobooth API",
    description="FastAPI Backend for Age Estimation, Receipt Printing, and QR Code Generation",
    version="1.0.0"
)

# -------------------------------------------------------------------
# 1. ONNX 나이 추정 모델 로드 (CPU Provider)
# -------------------------------------------------------------------
MODEL_PATH = "outputs/smoke/smoke_only.onnx"
ort_session: Optional[ort.InferenceSession] = None

try:
    # CPU 전용 추정 실행자 설정
    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print(f"[SUCCESS] ONNX Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[WARNING] Could not load ONNX model at startup: {e}")


def preprocess_image_for_onnx(image_bytes: bytes) -> np.ndarray:
    """전달받은 바이트 이미지를 ONNX 모델 입력 규격(1, 3, 224, 224)으로 전처리"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    
    # [0, 1] 정규화 및 NCHW 형태로 변환
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)    # CHW -> NCHW
    return arr


# -------------------------------------------------------------------
# 2. API Endpoints
# -------------------------------------------------------------------

@app.get("/")
def health_check():
    """서버 헬스체크"""
    return {
        "status": "healthy",
        "onnx_model_loaded": ort_session is not None,
        "device": "CPU"
    }


@app.post("/api/v1/predict-age")
async def predict_age(file: UploadFile = File(...)):
    """
    [1단계: 나이 추정]
    카메라로 입력받은 이미지를 전처리하여 ONNX 모델로 추정 나이(Age) 반환
    """
    if not ort_session:
        raise HTTPException(status_code=500, detail="ONNX Model session is not initialized.")
    
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image_for_onnx(image_bytes)
        
        # ONNX 추정 실행
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})
        
        # 추정 결과 파싱 (모델 output 형태에 맞게 조정)
        predicted_age = float(np.squeeze(outputs[0]))
        
        return {
            "status": "success",
            "filename": file.filename,
            "estimated_age": round(predicted_age, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


@app.get("/api/v1/generate-qr")
async def generate_qr(url: str, text: Optional[str] = "AI 타임머신이 보낸 당신의 미래 사진"):
    """
    [3단계: 영수증 프린터 연동용 QR 생성]
    고화질 사진 다운로드 URL을 받아 QR 코드를 생성하고, 하단에 텍스트를 합성한 1-bit Image 반환
    """
    try:
        # 1. QR 코드 생성 (흑백)
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        
        # 2. 영수증 인쇄 레이아웃 합성 (QR + 하단 텍스트)
        w, h = qr_img.size
        canvas_h = h + 40
        canvas = Image.new("RGB", (w, canvas_h), "white")
        canvas.paste(qr_img, (0, 0))
        
        draw = ImageDraw.Draw(canvas)
        # 기본 폰트 사용 (필요 시 custom 폰트로 교체)
        draw.text((w // 2, h + 10), text, fill="black", anchor="mm")
        
        # 3. PNG 바이트 반환
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QR generation error: {str(e)}")


@app.post("/api/v1/generate-timemachine-photo")
async def generate_timemachine_photo(
    file: UploadFile = File(...),
    target_age: int = Form(...),
    prompt: Optional[str] = Form("A photo of a person")
):
    """
    [2단계: 11/13 GPU 서버 연동을 위한 Dummy 생성 엔드포인트]
    GPU 서버 투입 전 백엔드 흐름 테스트용 (더미 결과 반환)
    """
    # 11/13 GPU 서버 준비 시 Diffusers / SD 3.5 파이프라인으로 교체되는 구역
    time.sleep(1.0) # 더미 추론 시간
    
    return {
        "status": "success",
        "target_age": target_age,
        "message": "GPU model pipeline placeholder. Will be updated on 11/13.",
        "download_url": f"https://trycloudflare.com/photos/dummy_{int(time.time())}.jpg"
    }