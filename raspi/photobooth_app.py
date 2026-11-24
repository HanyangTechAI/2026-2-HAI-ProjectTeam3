# booth-button-capture-v1
import threading
import uuid
import re
from pathlib import Path
from flask import redirect, url_for, send_from_directory, abort
from flask import Flask, request, render_template_string, Response
import serial
import time
import io
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera_lock = threading.Lock()
capture_busy = threading.Lock()
SHOT_DIR = Path.home() / "Pictures/ai_booth/web"
SHOT_DIR.mkdir(parents=True, exist_ok=True)

# 라이브 프리뷰용 해상도
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Time Machine Photo Booth</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; text-align: center; padding: 20px; background-color: #f2f2f7; color: #1c1c1e; }
        .card { background: white; padding: 20px; border-radius: 16px; max-width: 420px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.08); }
        h2 { margin-top: 0; font-size: 22px; color: #007aff; }
        .webcam-container { width: 100%; height: 260px; background: #000; border-radius: 12px; overflow: hidden; margin: 15px 0; }
        .webcam-container img { width: 100%; height: 100%; object-fit: cover; }
        textarea { width: 100%; height: 70px; font-size: 15px; padding: 10px; margin: 10px 0; border-radius: 10px; border: 1px solid #d1d1d6; box-sizing: border-box; resize: none; }
        button { font-size: 17px; font-weight: 600; width: 100%; padding: 14px; background-color: #007aff; color: white; border: none; border-radius: 12px; cursor: pointer; }
        .msg { margin-top: 15px; font-weight: bold; padding: 10px; border-radius: 8px; }
        .success { background-color: #e4f9e0; color: #2e7d32; }
        .error { background-color: #ffebee; color: #c62828; }
    </style>
</head>
<body>
    <div class="card">
        <h2>📸 AI 타임머신 포토부스</h2>
        
        <div class="webcam-container">
            <img src="/video_feed" alt="Webcam Feed">
        </div>

        <form method="POST" action="/capture_only" id="capture-form">
            <button type="submit">촬영하고 저장</button>
            <p id="capture-countdown" role="status" aria-live="polite">
                버튼을 누르면 3초 뒤 촬영해요. 인쇄하지 않아요.
            </p>
        </form>

        <form method="POST" action="/capture_and_print">
            <textarea name="caption" placeholder="하단 메시지 (선택)..."></textarea>
            <button type="submit">📷 촬영하고 인쇄하기</button>
        </form>

        {% if status %}
            <div class="msg {% if '성공' in status %}success{% else %}error{% endif %}">
                {{ status }}
            </div>
        {% endif %}
    </div>

    {% if captured_name %}
    <section style="max-width:420px;margin:20px auto;text-align:left;">
        <h3>촬영한 사진</h3>
        <img
            src="{{ url_for('saved_shot', filename=captured_name) }}"
            alt="방금 촬영한 사진"
            style="display:block;width:100%;height:auto;border-radius:10px;"
        >
        <p>
            <a href="{{ url_for('saved_shot', filename=captured_name, download=1) }}">
                사진 다운로드
            </a>
        </p>
        <p>사진은 라즈베리파이에 저장돼요. 자동으로 삭제되지 않아요.</p>
    </section>
    {% endif %}

    <script>
    const captureForm = document.getElementById("capture-form");
    let capturing = false;

    captureForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (capturing) return;
        capturing = true;

        document.querySelectorAll("button").forEach(button => {
            button.disabled = true;
        });

        const message = document.getElementById("capture-countdown");
        for (let count = 3; count >= 1; count--) {
            message.textContent = count + "초 뒤 촬영해요.";
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        message.textContent = "촬영하고 저장 중이에요.";
        captureForm.submit();
    });
    </script>
</body>
</html>
"""


# booth-horizontal-flip-v1
def read_camera_frame():
    success, frame = camera.read()
    if success and frame is not None:
        frame = cv2.flip(frame, 1)
    return success, frame


def generate_frames():
    while True:
        with camera_lock:
            success, frame = read_camera_frame()
        if not success:
            time.sleep(0.05)
            continue
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes() + b"\r\n"
        )

def wake_up_and_clear_printer(ser):
    ser.write(b'\x00\x00\x00')
    time.sleep(0.05)
    ser.write(b'\x1b\x40')
    time.sleep(0.05)
    ser.write(b'\x1b\x3d\x01')
    time.sleep(0.1)

def process_and_print_image(pil_img, caption=""):
    PRINTER_WIDTH = 384
    
    # 1. 384px 너비로 고화질 리사이즈 (LANCZOS 품질 필터)
    aspect_ratio = pil_img.height / pil_img.width
    target_height = int(PRINTER_WIDTH * aspect_ratio)
    img = pil_img.resize((PRINTER_WIDTH, target_height), Image.Resampling.LANCZOS)
    
    # 2. 흑백 변환
    img_gray = img.convert('L')
    
    # 3. Sharpening(선명도) 필터 적용 (이목구비 디테일 강조)
    img_sharp = img_gray.filter(ImageFilter.SHARPEN)
    
    # 4. 밝기 1.25배, 대조(Contrast) 1.3배 상승 (경계선이 뚜렷해짐)
    enhancer_bright = ImageEnhance.Brightness(img_sharp)
    img_bright = enhancer_bright.enhance(1.25)
    
    enhancer_contrast = ImageEnhance.Contrast(img_bright)
    img_enhanced = enhancer_contrast.enhance(1.3)
    
    # 5. Floyd-Steinberg 흑백 디더링 적용
    img_dithered = img_enhanced.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
    except:
        font = ImageFont.load_default()

    header_height = 45
    footer_height = 55 if caption else 35
    total_height = header_height + img_dithered.height + footer_height

    # 6. 레이아웃 합성
    final_img = Image.new('1', (PRINTER_WIDTH, total_height), 1)
    draw = ImageDraw.Draw(final_img)

    draw.text((20, 12), "AI TIME MACHINE PHOTO", fill=0, font=font)
    draw.line([(0, header_height - 5), (PRINTER_WIDTH, header_height - 5)], fill=0, width=2)

    final_img.paste(img_dithered, (0, header_height))

    footer_top = header_height + img_dithered.height
    draw.line([(0, footer_top + 5), (PRINTER_WIDTH, footer_top + 5)], fill=0, width=2)
    if caption:
        draw.text((20, footer_top + 15), caption, fill=0, font=font)

    # 7. 1비트 래스터 데이터 변환 (0x1D 0x76 0x30 0x00)
    img_byte_width = PRINTER_WIDTH // 8
    img_height = final_img.height

    header = bytearray([0x1D, 0x76, 0x30, 0x00, img_byte_width & 0xFF, (img_byte_width >> 8) & 0xFF, img_height & 0xFF, (img_height >> 8) & 0xFF])

    bitmap_data = bytearray()
    for py in range(img_height):
        for bx in range(img_byte_width):
            byte_val = 0
            for bit in range(8):
                px = bx * 8 + bit
                if px < PRINTER_WIDTH:
                    if final_img.getpixel((px, py)) == 0:
                        byte_val |= (1 << (7 - bit))
            bitmap_data.append(byte_val)

    # 8. 시리얼 안정적 청크 전송
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
    wake_up_and_clear_printer(ser)
    
    payload = header + bitmap_data
    CHUNK_SIZE = 1024
    for i in range(0, len(payload), CHUNK_SIZE):
        ser.write(payload[i:i+CHUNK_SIZE])
        time.sleep(0.01)

    ser.write(b'\n\n\n\n')
    ser.flush()
    ser.close()

@app.route('/')
def index():
    name = request.args.get("shot", "")
    if not re.fullmatch(r"[a-f0-9]{32}\.jpg", name):
        name = ""
    elif not (SHOT_DIR / name).is_file():
        name = ""
    status = "촬영 및 저장 성공! 인쇄하지 않았어요." if name else None
    return render_template_string(
        HTML_TEMPLATE, status=status, captured_name=name
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_print', methods=['POST'])
def capture_and_print():
    if not capture_busy.acquire(blocking=False):
        return render_template_string(
            HTML_TEMPLATE,
            status="다른 촬영·인쇄가 진행 중이에요. 잠시 후 다시 시도해 주세요.",
        ), 409
    try:
        caption = request.form.get("caption", "").strip()
        frame = capture_frame()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        process_and_print_image(pil_img, caption)
        status = "촬영 및 인쇄 성공!"
    except Exception:
        app.logger.exception("Capture or printing failed")
        status = "촬영 또는 인쇄에 실패했어요. 카메라와 프린터를 확인해 주세요."
    finally:
        capture_busy.release()
    return render_template_string(HTML_TEMPLATE, status=status)


def capture_frame():
    with camera_lock:
        try:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            frame = None
            success = False
            for _ in range(5):
                success, frame = read_camera_frame()
            if not success or frame is None:
                raise RuntimeError("Camera capture failed")
            return frame
        finally:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


@app.route("/capture_only", methods=["POST"])
def capture_only():
    if not capture_busy.acquire(blocking=False):
        return render_template_string(
            HTML_TEMPLATE,
            status="다른 촬영·인쇄가 진행 중이에요. 잠시 후 다시 시도해 주세요.",
        ), 409
    try:
        frame = capture_frame()
        name = uuid.uuid4().hex + ".jpg"
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        with (SHOT_DIR / name).open("xb") as file:
            file.write(encoded.tobytes())
        return redirect(url_for("index", shot=name), code=303)
    except Exception:
        app.logger.exception("Photo saving failed")
        return render_template_string(
            HTML_TEMPLATE,
            status="사진 저장에 실패했어요. 카메라 연결과 저장 공간을 확인해 주세요.",
        ), 500
    finally:
        capture_busy.release()


@app.route("/shots/<filename>")
def saved_shot(filename):
    if not re.fullmatch(r"[a-f0-9]{32}\.jpg", filename):
        abort(404)
    response = send_from_directory(
        str(SHOT_DIR),
        filename,
        as_attachment=request.args.get("download") == "1",
        mimetype="image/jpeg",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
