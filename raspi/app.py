from flask import Flask, request, render_template_string, Response
import serial
import time
import io
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

app = Flask(__name__)
camera = cv2.VideoCapture(0)

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
</body>
</html>
"""

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_print', methods=['POST'])
def capture_and_print():
    caption = request.form.get('caption', '').strip()
    
    # 캡처 순간 고해상도(1920x1080) 전환 후 캡처
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    for _ in range(3):
        camera.grab()
    success, frame = camera.read()
    
    # 프리뷰 해상도로 복원
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if success:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        try:
            process_and_print_image(pil_img, caption)
            status = "✅ 촬영 및 인쇄 성공!"
        except Exception as e:
            status = f"❌ 인쇄 에러: {e}"
    else:
        status = "❌ 웹캠 캡처 실패!"

    return render_template_string(HTML_TEMPLATE, status=status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
