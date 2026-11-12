import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response, FileResponse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import qrcode
import onnxruntime as ort

from gen.booth import run_booth, compose_four_cut, MockAgeEstimator
from gen.pipeline import PipelineConfig, AgeShiftPipeline, MockPipeline

app = FastAPI(
    title="AI Timemachine Photobooth API",
    description="FastAPI Backend for Age Estimation, 4-Cut Timemachine Generation, and Receipt QR",
    version="2.0.0"
)

OUTPUT_PHOTOS_DIR = Path("outputs/photos")
OUTPUT_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

KOREAN_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf"
MODEL_PATH = "outputs/smoke/smoke_only.onnx"

ort_session: Optional[ort.InferenceSession] = None
try:
    ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f"[SUCCESS] ONNX Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[WARNING] Could not load ONNX model: {e}")

USE_MOCK_GEN = os.getenv("USE_MOCK_GEN", "true").lower() == "true"
GEN_DEVICE = os.getenv("GEN_DEVICE", "cpu")

pipeline_config = PipelineConfig(
    base_model="" if USE_MOCK_GEN else os.getenv("BASE_MODEL_PATH", ""),
    device=GEN_DEVICE,
)
booth_generator = MockPipeline(mode="faceid_age") if USE_MOCK_GEN else AgeShiftPipeline(pipeline_config, mode="faceid_age")


def preprocess_image_for_onnx(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def estimate_age_from_pil(pil_img: Image.Image) -> float:
    if not ort_session:
        return 25.0
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: arr})
    return round(float(np.squeeze(outputs[0])), 1)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "onnx_model_loaded": ort_session is not None,
        "mock_generation_mode": USE_MOCK_GEN,
        "device": GEN_DEVICE
    }


@app.post("/api/v1/predict-age")
async def predict_age(file: UploadFile = File(...)):
    if not ort_session:
        raise HTTPException(status_code=500, detail="ONNX Model session is not initialized.")
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image_for_onnx(image_bytes)
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})
        predicted_age = float(np.squeeze(outputs[0]))
        return {
            "status": "success",
            "filename": file.filename,
            "estimated_age": round(predicted_age, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


@app.post("/api/v1/generate-timemachine-photo")
async def generate_timemachine_photo(
    request: Request,
    file: UploadFile = File(...),
    target_age: Optional[int] = Form(None),
    seed: int = Form(42)
):
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        current_age = float(target_age) if target_age is not None else estimate_age_from_pil(pil_img)
        estimator = MockAgeEstimator(age=current_age)

        result = run_booth(pil_img, estimator, booth_generator, seed=seed)
        four_cut_sheet = compose_four_cut(result["frames"], cell_size=(512, 512))

        filename = f"timemachine_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        save_path = OUTPUT_PHOTOS_DIR / filename
        four_cut_sheet.save(save_path, format="PNG")

        base_url = str(request.base_url).rstrip("/")
        if request.headers.get("x-forwarded-proto") == "https" and base_url.startswith("http://"):
            base_url = "https://" + base_url[len("http://"):]

        download_url = f"{base_url}/photos/{filename}"

        return {
            "status": "success",
            "estimated_age": current_age,
            "targets": result["metadata"]["targets"],
            "is_mock": result["metadata"]["is_mock"],
            "filename": filename,
            "download_url": download_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timemachine generation error: {str(e)}")


@app.get("/photos/{filename}")
async def download_photo(filename: str):
    safe_name = Path(filename).name
    file_path = OUTPUT_PHOTOS_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(path=file_path, media_type="image/png", filename=safe_name)


@app.get("/api/v1/generate-qr")
async def generate_qr(url: str, text: Optional[str] = "AI 타임머신이 보낸 당신의 미래 사진"):
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        w, h = qr_img.size
        canvas_h = h + 48
        canvas = Image.new("RGB", (w, canvas_h), "white")
        canvas.paste(qr_img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        if Path(KOREAN_FONT_PATH).is_file():
            font = ImageFont.truetype(KOREAN_FONT_PATH, size=15)
        else:
            font = ImageFont.load_default()

        draw.text((w // 2, h + 20), text, fill="black", font=font, anchor="mm")

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QR generation error: {str(e)}")
