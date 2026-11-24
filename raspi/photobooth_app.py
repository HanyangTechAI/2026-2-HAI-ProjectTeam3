# booth-timetravel-ui-v1
# 촬영 → FADING 4컷 생성(백그라운드) → 상태 폴링 → 결과+QR 표시.
# 기존 원본 인쇄 기능은 그대로 보존.
import io
import re
import sys
import time
import uuid
import socket
import threading
from pathlib import Path

import cv2
import requests
import qrcode
import serial
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from flask import (
    Flask, request, render_template_string, Response,
    jsonify, send_from_directory, abort,
)

app = Flask(__name__)

# ===================== 설정 (환경에 맞게 수정) =====================
# 라즈베리파이 → (맥 SSH 중계) → HAI 로 가는 터널의 라즈베리파이 쪽 주소.
# 인수인계 문서 기준: 라즈베리파이 18000 → HAI 8000. 실제 포트에 맞게 바꿔.
FADING_API_URL = "http://127.0.0.1:18000/api/fading/fourcut"
API_TIMEOUT = 300          # FADING 1회 ~100초. 여유있게 5분.
WEB_PORT = 5000
# QR이 가리킬 주소(핸드폰이 접속). 라즈베리파이 LAN IP 자동 감지, 필요시 직접 지정.
def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()
PUBLIC_HOST = _lan_ip()
# ================================================================

BASE_DIR = Path.home() / "Pictures/ai_booth"
CAPTURE_DIR = BASE_DIR / "capture"
RESULT_DIR = BASE_DIR / "result"
QR_DIR = BASE_DIR / "qr"
for d in (CAPTURE_DIR, RESULT_DIR, QR_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 윈도우 노트북에서 테스트할 땐 DirectShow가 해상도 전환에 더 안정적. 라즈베리파이는 기본값.
camera = (cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win")
          else cv2.VideoCapture(0))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera_lock = threading.Lock()
capture_busy = threading.Lock()   # 중복 촬영 방지

# 진행 중/완료된 작업 상태 저장소
jobs = {}
jobs_lock = threading.Lock()
ID_RE = re.compile(r"[a-f0-9]{32}")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 타임머신 포토부스</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         text-align:center; padding:20px; background:#f2f2f7; color:#1c1c1e; margin:0; }
  .card { background:#fff; padding:22px; border-radius:18px; max-width:440px; margin:0 auto;
          box-shadow:0 10px 25px rgba(0,0,0,.08); }
  h2 { margin:0 0 6px; font-size:23px; color:#007aff; }
  .sub { color:#8e8e93; font-size:14px; margin-bottom:14px; }
  .stage { width:100%; aspect-ratio:4/3; background:#000; border-radius:14px; overflow:hidden;
           margin:14px 0; position:relative; }
  .stage img, .stage video { width:100%; height:100%; object-fit:cover; display:block; }
  /* 영상 위 카운트다운 오버레이 */
  #countdown { position:absolute; inset:0; display:none; align-items:center; justify-content:center;
               font-size:120px; font-weight:800; color:#fff; text-shadow:0 2px 16px rgba(0,0,0,.6);
               background:rgba(0,0,0,.25); }
  .gender { display:flex; gap:10px; margin:8px 0 14px; }
  .gender button { flex:1; padding:12px; border-radius:12px; border:2px solid #d1d1d6;
                   background:#fff; color:#1c1c1e; font-size:16px; font-weight:600; cursor:pointer; }
  .gender button.on { border-color:#007aff; background:#eaf3ff; color:#007aff; }
  .primary { font-size:19px; font-weight:700; width:100%; padding:16px; background:#007aff;
             color:#fff; border:none; border-radius:14px; cursor:pointer; }
  .primary:disabled { background:#b9b9be; cursor:default; }
  .ghost { margin-top:10px; background:none; border:none; color:#8e8e93; font-size:14px;
           text-decoration:underline; cursor:pointer; }
  .msg { margin-top:14px; font-weight:700; padding:12px; border-radius:10px; }
  .error { background:#ffebee; color:#c62828; }
  .spin { width:42px; height:42px; border:4px solid #d1d1d6; border-top-color:#007aff;
          border-radius:50%; margin:18px auto; animation:spin 1s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .elapsed { font-size:28px; font-weight:800; color:#007aff; }
  .qr { width:180px; height:180px; margin:12px auto; display:block; }
  textarea { width:100%; height:56px; font-size:15px; padding:10px; margin:6px 0 2px;
             border-radius:10px; border:1px solid #d1d1d6; box-sizing:border-box; resize:none; }
  .hidden { display:none !important; }
</style>
</head>
<body>
<div class="card">
  <h2>📸 AI 타임머신 포토부스</h2>

  <!-- 1) 대기 화면 -->
  <section id="idle">
    <div class="sub">성별을 고르고 버튼을 누르면 시간 여행이 시작돼요.</div>
    <div class="stage">
      <img src="/video_feed" alt="웹캠">
      <div id="countdown"></div>
    </div>
    <div class="gender">
      <button type="button" data-g="male"   onclick="pickGender('male')">남성</button>
      <button type="button" data-g="female" onclick="pickGender('female')">여성</button>
    </div>
    <button id="startBtn" class="primary" onclick="startTimeTravel()">⏳ 시간 여행 시작</button>

    <!-- 기존 원본 인쇄 기능 보존 -->
    <form method="POST" action="/capture_and_print" style="margin-top:16px;">
      <textarea name="caption" placeholder="원본 영수증 인쇄용 메시지 (선택)…"></textarea>
      <button type="submit" class="ghost">원본 사진만 인쇄하기</button>
    </form>
    {% if status %}<div class="msg {% if '성공' in status %}success{% else %}error{% endif %}">{{ status }}</div>{% endif %}
  </section>

  <!-- 2) 처리 중 화면 -->
  <section id="processing" class="hidden">
    <div class="spin"></div>
    <div style="font-weight:700;font-size:18px;">시간 여행 중이에요…</div>
    <div class="sub">과거와 미래의 얼굴을 그리고 있어요. 1~2분 걸려요.</div>
    <div class="elapsed" id="elapsed">00:00</div>
  </section>

  <!-- 3) 완료 화면 -->
  <section id="done" class="hidden">
    <h2>🎉 시간 여행 완료!</h2>
    <div class="sub" id="ageNote"></div>
    <div class="stage" style="aspect-ratio:auto;background:#fff;">
      <img id="resultImg" alt="4컷 결과" style="object-fit:contain;height:auto;">
    </div>
    <div class="sub">QR을 찍으면 고화질 사진을 받을 수 있어요.</div>
    <img id="qrImg" class="qr" alt="다운로드 QR">
    <button class="primary" onclick="location.reload()">다시 하기</button>
  </section>

  <!-- 4) 실패 화면 -->
  <section id="failed" class="hidden">
    <div class="msg error" id="errMsg">문제가 생겼어요.</div>
    <button class="primary" onclick="location.reload()">다시 시도</button>
  </section>
</div>

<script>
let gender = null;
let busy = false;

function pickGender(g) {
  gender = g;
  document.querySelectorAll(".gender button").forEach(b =>
    b.classList.toggle("on", b.dataset.g === g));
}
function show(id) {
  ["idle","processing","done","failed"].forEach(s =>
    document.getElementById(s).classList.toggle("hidden", s !== id));
}
function fmt(sec) {
  const m = String(Math.floor(sec/60)).padStart(2,"0");
  const s = String(sec%60).padStart(2,"0");
  return m+":"+s;
}

async function startTimeTravel() {
  if (busy) return;
  if (!gender) { alert("성별을 먼저 골라주세요."); return; }
  busy = true;
  document.getElementById("startBtn").disabled = true;

  // 영상 위 3·2·1 카운트다운
  const cd = document.getElementById("countdown");
  cd.style.display = "flex";
  for (let n = 3; n >= 1; n--) { cd.textContent = n; await new Promise(r => setTimeout(r, 1000)); }
  cd.textContent = "📸"; await new Promise(r => setTimeout(r, 400));
  cd.style.display = "none";

  // 촬영 + 작업 시작
  let jobId;
  try {
    const fd = new FormData();
    fd.append("gender", gender);
    const res = await fetch("/start_time_travel", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "촬영에 실패했어요.");
    jobId = data.job_id;
  } catch (e) {
    return fail(e.message);
  }

  // 처리 중 화면 + 경과 시간
  show("processing");
  let elapsed = 0;
  const timer = setInterval(() => {
    elapsed++; document.getElementById("elapsed").textContent = fmt(elapsed);
  }, 1000);

  // 상태 폴링 (2초마다)
  const poll = setInterval(async () => {
    try {
      const r = await fetch("/status/" + jobId);
      const s = await r.json();
      if (s.status === "done") {
        clearInterval(poll); clearInterval(timer);
        document.getElementById("resultImg").src = "/result/" + jobId + "?t=" + Date.now();
        document.getElementById("qrImg").src = "/qr/" + jobId + "?t=" + Date.now();
        if (s.age) document.getElementById("ageNote").textContent = "추정 나이 약 " + s.age + "세";
        show("done");
      } else if (s.status === "error") {
        clearInterval(poll); clearInterval(timer);
        fail(s.error || "생성에 실패했어요.");
      }
    } catch (e) { /* 일시적 네트워크 오류는 무시하고 계속 폴링 */ }
  }, 2000);
}

function fail(message) {
  document.getElementById("errMsg").textContent = message;
  show("failed");
}
</script>
</body>
</html>
"""


# ===================== 카메라 =====================
def read_camera_frame():
    success, frame = camera.read()
    if success and frame is not None:
        frame = cv2.flip(frame, 1)   # 좌우반전(거울)
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
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


def _grab(tries):
    """여러 장 읽어서 마지막으로 성공한 프레임을 돌려준다. 해상도 변경 직후 빈 프레임 대비."""
    frame = None
    for _ in range(tries):
        success, f = read_camera_frame()
        if success and f is not None:
            frame = f
        else:
            time.sleep(0.05)
    return frame


def capture_frame():
    with camera_lock:
        try:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            frame = _grab(20)
            if frame is None:
                # 고해상도를 지원하지 않는 웹캠이면 미리보기 해상도로 다시 촬영
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                frame = _grab(10)
            if frame is None:
                raise RuntimeError("Camera capture failed")
            return frame
        finally:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ===================== FADING 백그라운드 작업 =====================
def _set_job(job_id, **kw):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(kw)


def run_fading_job(job_id, image_path, gender):
    """촬영 사진을 HAI FADING API로 보내 4컷을 받고, 결과·QR을 저장한다."""
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                FADING_API_URL,
                files={"file": ("input.jpg", f, "image/jpeg")},
                data={"gender": gender},
                timeout=API_TIMEOUT,
            )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = ""
            _set_job(job_id, status="error",
                     error=detail or f"생성 실패 (코드 {resp.status_code})")
            return

        # 4컷 PNG 저장
        (RESULT_DIR / f"{job_id}.png").write_bytes(resp.content)

        # 결과 다운로드 URL로 QR 생성 (핸드폰이 같은 네트워크에서 접속)
        url = f"http://{PUBLIC_HOST}:{WEB_PORT}/result/{job_id}?download=1"
        qrcode.make(url).save(QR_DIR / f"{job_id}.png")

        age = resp.headers.get("X-Estimated-Age", "")
        _set_job(job_id, status="done", age=age, url=url)
    except requests.Timeout:
        _set_job(job_id, status="error", error="생성 시간이 초과됐어요. 다시 시도해 주세요.")
    except requests.RequestException:
        _set_job(job_id, status="error", error="서버에 연결하지 못했어요. 연결을 확인해 주세요.")
    except Exception:
        app.logger.exception("FADING job failed")
        _set_job(job_id, status="error", error="알 수 없는 오류가 났어요. 다시 시도해 주세요.")


# ===================== 라우트 =====================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, status=None)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_time_travel", methods=["POST"])
def start_time_travel():
    gender = request.form.get("gender", "")
    if gender not in ("male", "female"):
        return jsonify(error="성별을 선택해 주세요."), 400
    if not capture_busy.acquire(blocking=False):
        return jsonify(error="다른 촬영이 진행 중이에요. 잠시 후 다시 시도해 주세요."), 409
    try:
        # HAI는 한 번에 한 장만 처리하므로, 생성 중인 작업이 있으면 새 촬영을 받지 않는다.
        with jobs_lock:
            running = any(j.get("status") == "processing" for j in jobs.values())
        if running:
            return jsonify(error="지금 다른 분의 시간 여행을 만드는 중이에요. 끝나면 다시 시도해 주세요."), 409
        frame = capture_frame()
        job_id = uuid.uuid4().hex
        path = CAPTURE_DIR / f"{job_id}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        path.write_bytes(encoded.tobytes())

        _set_job(job_id, status="processing", started=time.time())
        threading.Thread(target=run_fading_job, args=(job_id, str(path), gender),
                         daemon=True).start()
        return jsonify(job_id=job_id)
    except Exception:
        app.logger.exception("capture failed")
        return jsonify(error="촬영에 실패했어요. 카메라를 확인해 주세요."), 500
    finally:
        capture_busy.release()


@app.route("/status/<job_id>")
def status(job_id):
    if not ID_RE.fullmatch(job_id):
        abort(404)
    with jobs_lock:
        job = dict(jobs.get(job_id, {}))
    if not job:
        abort(404)
    return jsonify(status=job.get("status", "processing"),
                   age=job.get("age", ""), error=job.get("error", ""))


@app.route("/result/<job_id>")
def result(job_id):
    if not ID_RE.fullmatch(job_id):
        abort(404)
    return send_from_directory(str(RESULT_DIR), f"{job_id}.png",
                               as_attachment=request.args.get("download") == "1",
                               mimetype="image/png")


@app.route("/qr/<job_id>")
def qr(job_id):
    if not ID_RE.fullmatch(job_id):
        abort(404)
    return send_from_directory(str(QR_DIR), f"{job_id}.png", mimetype="image/png")


# ---------- 기존 원본 인쇄 기능 (보존) ----------
def wake_up_and_clear_printer(ser):
    ser.write(b"\x00\x00\x00"); time.sleep(0.05)
    ser.write(b"\x1b\x40"); time.sleep(0.05)
    ser.write(b"\x1b\x3d\x01"); time.sleep(0.1)


def process_and_print_image(pil_img, caption=""):
    PRINTER_WIDTH = 384
    aspect = pil_img.height / pil_img.width
    img = pil_img.resize((PRINTER_WIDTH, int(PRINTER_WIDTH * aspect)), Image.Resampling.LANCZOS)
    img = img.convert("L").filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Brightness(img).enhance(1.25)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    header_h = 45
    footer_h = 55 if caption else 35
    final_img = Image.new("1", (PRINTER_WIDTH, header_h + img.height + footer_h), 1)
    draw = ImageDraw.Draw(final_img)
    draw.text((20, 12), "AI TIME MACHINE PHOTO", fill=0, font=font)
    draw.line([(0, header_h - 5), (PRINTER_WIDTH, header_h - 5)], fill=0, width=2)
    final_img.paste(img, (0, header_h))
    footer_top = header_h + img.height
    draw.line([(0, footer_top + 5), (PRINTER_WIDTH, footer_top + 5)], fill=0, width=2)
    if caption:
        draw.text((20, footer_top + 15), caption, fill=0, font=font)

    byte_w = PRINTER_WIDTH // 8
    header = bytearray([0x1D, 0x76, 0x30, 0x00, byte_w & 0xFF, (byte_w >> 8) & 0xFF,
                        final_img.height & 0xFF, (final_img.height >> 8) & 0xFF])
    bitmap = bytearray()
    for py in range(final_img.height):
        for bx in range(byte_w):
            val = 0
            for bit in range(8):
                px = bx * 8 + bit
                if px < PRINTER_WIDTH and final_img.getpixel((px, py)) == 0:
                    val |= (1 << (7 - bit))
            bitmap.append(val)
    ser = serial.Serial("/dev/ttyACM0", 115200, timeout=2)
    wake_up_and_clear_printer(ser)
    payload = header + bitmap
    for i in range(0, len(payload), 1024):
        ser.write(payload[i:i+1024]); time.sleep(0.01)
    ser.write(b"\n\n\n\n"); ser.flush(); ser.close()


@app.route("/capture_and_print", methods=["POST"])
def capture_and_print():
    if not capture_busy.acquire(blocking=False):
        return render_template_string(
            HTML_TEMPLATE, status="다른 촬영이 진행 중이에요. 잠시 후 다시 시도해 주세요."), 409
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


if __name__ == "__main__":
    print(f"QR 접속 주소 기준 호스트: {PUBLIC_HOST}:{WEB_PORT}  (틀리면 PUBLIC_HOST 수정)")
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)
