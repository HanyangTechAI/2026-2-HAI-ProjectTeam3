
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.age_service import RealAgeEstimator

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path.home() / ".cache/booth-auto"
FADING_RESULTS = Path.home() / ".cache/fading-results"
ROOT.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
TARGETS = [10, 30, 50]

app = FastAPI(title="FADING Photobooth", docs_url=None, redoc_url=None)
busy = Lock()
logger = logging.getLogger(__name__)

print("実モデル".replace("実モデル", "실제 나이 추정 모델 로딩 중..."), flush=True)
estimator = RealAgeEstimator()
print("자동 4컷 요청을 받을 준비가 됐어요.", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "photobooth-fading",
        "real_age_model": True,
        "targets": TARGETS,
    }


@app.post("/api/fading/fourcut")
def make_fourcut(
    file: UploadFile = File(...),
    gender: str = Form(...),
):
    if gender not in ("female", "male"):
        raise HTTPException(400, "성별 조건을 선택해 주세요.")

    if not busy.acquire(blocking=False):
        raise HTTPException(409, "다른 사진을 처리 중이에요.")

    started = time.perf_counter()
    job_id = uuid.uuid4().hex
    folder = ROOT / job_id

    try:
        if shutil.disk_usage(ROOT).free < 512 * 1024**2:
            raise HTTPException(507, "HAI 저장 공간이 부족해요.")

        data = file.file.read(MAX_BYTES + 1)
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(400, "사진은 10MB 이하로 보내주세요.")

        try:
            with Image.open(io.BytesIO(data)) as opened:
                if (
                    min(opened.size) < 32
                    or opened.width * opened.height > MAX_PIXELS
                ):
                    raise ValueError("Unsupported image dimensions")
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
        except Exception:
            raise HTTPException(400, "사진 파일을 읽지 못했어요.")

        try:
            result = estimator(image)
            predicted_age = float(result["age"])
        except Exception:
            logger.exception("Age estimation failed")
            raise HTTPException(
                422,
                "얼굴 한 명이 선명하게 나오도록 다시 촬영해 주세요.",
            )

        initial_age = max(1, min(100, int(predicted_age + 0.5)))

        folder.mkdir(exist_ok=False)
        source = folder / "input.jpg"
        image.save(source, quality=95)
        log_path = folder / "generation.log"

        print(
            f"[{job_id}] estimated={predicted_age:.1f}, "
            f"initial={initial_age}, gender={gender}",
            flush=True,
        )

        command = [
            sys.executable,
            str(PROJECT / "scripts/fading_infer.py"),
            "--image", str(source),
            "--age", str(initial_age),
            "--gender", gender,
            "--targets", *map(str, TARGETS),
            "--self-replace", "0.9",
        ]

        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=PROJECT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^Results: (.+)$", log_text, re.MULTILINE)
        if match is None:
            raise RuntimeError("FADING output directory not reported")

        generated = Path(match.group(1).strip()).resolve()
        if not generated.is_relative_to(FADING_RESULTS.resolve()):
            raise RuntimeError("Unexpected output directory")

        sys.path.insert(0, str(Path.home() / ".cache/fading-src"))
        from null_inversion import load_512

        tiles = [
            ("Original", Image.fromarray(load_512(str(source))))
        ]
        for age in TARGETS:
            with Image.open(generated / f"input_{age}.png") as frame:
                tiles.append((
                    f"Target age {age}",
                    frame.convert("RGB").copy(),
                ))

        cell, header = 512, 44
        sheet = Image.new(
            "RGB", (cell * 2, (cell + header) * 2), "white"
        )
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()

        for index, (label, frame) in enumerate(tiles):
            x = (index % 2) * cell
            y = (index // 2) * (cell + header)
            draw.text(
                (x + 14, y + 10), label, fill="black", font=font
            )
            sheet.paste(
                frame.resize((cell, cell)), (x, y + header)
            )

        output = folder / "fourcut.png"
        sheet.save(output)
        seconds = time.perf_counter() - started

        metadata = {
            "job_id": job_id,
            "estimated_age": predicted_age,
            "initial_age": initial_age,
            "age_source": "real_model",
            "gender_condition": gender,
            "targets": TARGETS,
            "self_replace_steps": 0.9,
            "total_seconds": seconds,
            "generation_is_mock": False,
        }
        (folder / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[{job_id}] completed in {seconds:.1f}s", flush=True)

        return FileResponse(
            output,
            media_type="image/png",
            filename=f"fourcut_{job_id}.png",
            headers={
                "Cache-Control": "no-store",
                "X-Estimated-Age": f"{predicted_age:.1f}",
                "X-Processing-Seconds": f"{seconds:.1f}",
                "X-Job-Id": job_id,
            },
        )

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        logger.exception("FADING timed out: %s", job_id)
        raise HTTPException(504, "생성 시간이 초과됐어요.")
    except Exception:
        logger.exception("Fourcut generation failed: %s", job_id)
        raise HTTPException(
            500,
            "4컷 생성에 실패했어요. HAI 처리 기록을 확인해 주세요.",
        )
    finally:
        busy.release()
