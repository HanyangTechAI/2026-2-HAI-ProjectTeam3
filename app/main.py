import io
import logging
import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
)
from fastapi.responses import Response, FileResponse
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
import qrcode

from app.age_service import RealAgeEstimator
from gen.booth import run_booth, compose_four_cut, MockAgeEstimator
from gen.pipeline import PipelineConfig, AgeShiftPipeline, MockPipeline


logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 타임머신 포토부스",
    description="실제 나이 추정 및 4컷 포토부스 테스트",
    version="2.1.0",
)

OUTPUT_PHOTOS_DIR = Path("outputs/photos")
OUTPUT_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

KOREAN_FONT_PATH = (
    "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf"
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

# 실패하면 시작을 중단해요. 임의의 나이로 대체하지 않아요.
age_estimator = RealAgeEstimator()
age_lock = Lock()
generation_lock = Lock()

# M1 연결 전까지 기본값은 모의 생성이에요.
USE_MOCK_GEN = os.getenv("USE_MOCK_GEN", "true").lower() == "true"
GEN_DEVICE = os.getenv("GEN_DEVICE", "cpu")

pipeline_config = PipelineConfig(
    base_model="" if USE_MOCK_GEN else os.getenv("BASE_MODEL_PATH", ""),
    device=GEN_DEVICE,
)

if USE_MOCK_GEN:
    booth_generator = MockPipeline(mode="faceid_age")
else:
    # 기존 생성 경로예요. 앞으로 학습할 M1과는 별개예요.
    booth_generator = AgeShiftPipeline(
        pipeline_config,
        mode="faceid_age",
    )


def read_uploaded_image(file: UploadFile) -> Image.Image:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="사진 파일이 비어 있어요.",
        )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="사진이 너무 커요. 10MB 이하로 보내주세요.",
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "MPO"}:

                raise HTTPException(
                    status_code=415,
                    detail="JPG 또는 PNG 사진을 보내주세요.",
                )

            width, height = image.size
            if min(width, height) < 16:
                raise HTTPException(
                    status_code=422,
                    detail="사진이 너무 작아요.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="사진 해상도가 너무 커요. 크기를 줄여주세요.",
                )
            image.seek(0)
            return ImageOps.exif_transpose(image).convert("RGB")

    except HTTPException:
        raise
    except Image.DecompressionBombError as error:
        raise HTTPException(
            status_code=413,
            detail="사진 해상도가 너무 커요.",
        ) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="사진을 읽지 못했어요. 정상적인 JPG 또는 PNG를 보내주세요.",
        ) from error


def estimate_age_from_pil(image: Image.Image) -> float:
    try:
        with age_lock:
            result = age_estimator(image)

        return float(result["age"])

    except ValueError as error:
        messages = {
            "face_not_detected": "얼굴을 찾지 못했어요. 정면으로 다시 촬영해 주세요.",
            "multiple_faces": "얼굴이 여러 개예요. 한 명씩 촬영해 주세요.",
            "image_too_small": "사진이 너무 작아요.",
            "invalid_landmarks": "얼굴 위치를 확인하지 못했어요.",
            "invalid_similarity_transform": "얼굴 위치를 처리하지 못했어요.",
        }

        if str(error) in messages:
            raise HTTPException(
                status_code=422,
                detail=messages[str(error)],
            ) from error

        logger.exception("나이 추정 중 잘못된 결과가 발생했어요.")
        raise HTTPException(
            status_code=500,
            detail="나이 추정에 실패했어요.",
        ) from error

    except Exception as error:
        logger.exception("나이 추정 중 오류가 발생했어요.")
        raise HTTPException(
            status_code=500,
            detail="나이 추정에 실패했어요.",
        ) from error


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "age_model_loaded": True,
        "age_model": "public/best.pt",
        "age_device": age_estimator.device,
        "mock_generation_mode": USE_MOCK_GEN,
        "generation_device": GEN_DEVICE,
    }


@app.post("/api/v1/predict-age")
def predict_age(file: UploadFile = File(...)):
    image = read_uploaded_image(file)
    predicted_age = estimate_age_from_pil(image)

    return {
        "status": "success",
        "estimated_age": round(predicted_age, 1),
        "is_mock": False,
    }


@app.post("/api/v1/generate-timemachine-photo")
def generate_timemachine_photo(
    request: Request,
    file: UploadFile = File(...),
    target_age: Optional[int] = Form(
        None,
        ge=0,
        le=100,
        description=(
            "기존 연결 호환용 현재 나이 수동 지정값이에요. "
            "변환할 목표 나이가 아니며, 생략하면 추정 나이를 사용해요."
        ),
    ),
    seed: int = Form(42),
):
    image = read_uploaded_image(file)

    if not generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="다른 사진을 처리 중이에요. 완료 후 다시 시도해 주세요.",
        )

    try:
        # 수동 나이를 지정해도 얼굴 검사와 실제 추정은 수행해요.
        predicted_age = estimate_age_from_pil(image)

        if target_age is not None:
            current_age = float(target_age)
            estimator = MockAgeEstimator(age=current_age)
            age_source = "manual_override"
        else:
            current_age = predicted_age
            age_source = "model"

            # run_booth에 실제 추정 결과를 전달해요.
            def estimator(_image):
                return {
                    "age": predicted_age,
                    "gender": "unknown",
                    "is_mock": False,
                }

        result = run_booth(
            image,
            estimator,
            booth_generator,
            seed=seed,
        )
        sheet = compose_four_cut(
            result["frames"],
            cell_size=(512, 512),
        )

        filename = (
            f"timemachine_{int(time.time())}_"
            f"{uuid.uuid4().hex[:12]}.png"
        )
        sheet.save(OUTPUT_PHOTOS_DIR / filename, format="PNG")

        download_url = str(
            request.url_for("download_photo", filename=filename)
        )

        metadata = result["metadata"]

        return {
            "status": "success",
            "estimated_age": round(predicted_age, 1),
            "current_age": round(current_age, 1),
            "age_source": age_source,
            "targets": metadata["targets"],
            "is_mock": metadata["is_mock"],
            "age_is_mock": metadata["age_is_mock"],
            "generation_is_mock": metadata["generation_is_mock"],
            "filename": filename,
            "download_url": download_url,
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("4컷 생성 중 오류가 발생했어요.")
        raise HTTPException(
            status_code=500,
            detail="4컷 생성에 실패했어요.",
        ) from error
    finally:
        generation_lock.release()


@app.get("/photos/{filename}", name="download_photo")
def download_photo(filename: str):
    if (
        Path(filename).name != filename
        or not filename.startswith("timemachine_")
        or not filename.endswith(".png")
    ):
        raise HTTPException(
            status_code=404,
            detail="사진을 찾지 못했어요.",
        )

    file_path = OUTPUT_PHOTOS_DIR / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="사진을 찾지 못했어요.",
        )

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=filename,
    )


@app.get("/api/v1/generate-qr")
def generate_qr(
    url: str,
    text: str = "AI 타임머신이 보낸 당신의 미래 사진",
):
    if not url or len(url) > 2000 or len(text) > 100:
        raise HTTPException(
            status_code=400,
            detail="주소나 문구가 비어 있거나 너무 길어요.",
        )

    try:
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)

        qr_image = qr.make_image(
            fill_color="black",
            back_color="white",
        ).convert("RGB")

        width, height = qr_image.size
        canvas = Image.new(
            "RGB",
            (width, height + 48),
            "white",
        )
        canvas.paste(qr_image, (0, 0))

        if Path(KOREAN_FONT_PATH).is_file():
            font = ImageFont.truetype(KOREAN_FONT_PATH, size=15)
        else:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(canvas)
        draw.text(
            (width // 2, height + 20),
            text,
            fill="black",
            font=font,
            anchor="mm",
        )

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
        )

    except Exception as error:
        logger.exception("QR 생성 중 오류가 발생했어요.")
        raise HTTPException(
            status_code=500,
            detail="QR 생성에 실패했어요.",
        ) from error
