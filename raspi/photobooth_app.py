# booth-timetravel-ui-v1
# 촬영 → FADING 4컷 생성(백그라운드) → 상태 폴링 → 결과+QR 표시.
# 기존 원본 인쇄 기능은 그대로 보존.
import io
import os
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
QR_BOX_SIZE = 10
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
PUBLIC_HOST = os.environ.get("PUBLIC_HOST") or _lan_ip()
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
printer_busy = threading.Lock()

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
  #flash { position:absolute; inset:0; background:#fff; opacity:0;
           transition:opacity .25s; pointer-events:none; }
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
      <div id="flash"></div>
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
    <div class="sub">10년 전 → 현재 → 10년 후 → 20년 후의 모습을 만들고 있어요. 편집 나이는 3~90세 범위이며 실제 과거나 미래를 보여 주지는 않아요.</div>
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
    <button id="printBtn" class="primary" disabled>4컷과 QR 인쇄하기</button>
    <div id="printMsg" role="status"></div>
    <button class="ghost" onclick="location.reload()">다시 하기</button>
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

  // 셔터: 실제 촬영이 끝날 때까지 "움직이지 마세요"를 유지한다
  cd.style.flexDirection = "column";
  cd.innerHTML = '<div style="font-size:72px">📸</div>' +
                 '<div style="font-size:22px;margin-top:6px">움직이지 마세요</div>';

  // 촬영 + 작업 시작 (응답이 오면 사진은 이미 찍힌 상태)
  let jobId;
  try {
    const fd = new FormData();
    fd.append("gender", gender);
    const res = await fetch("/start_time_travel", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "촬영에 실패했어요.");
    jobId = data.job_id;
  } catch (e) {
    cd.style.display = "none";
    return fail(e.message);
  }

  // 찍힌 순간 화면을 하얗게 번쩍
  cd.style.display = "none";
  const flash = document.getElementById("flash");
  flash.style.opacity = "1"; await new Promise(r => setTimeout(r, 120));
  flash.style.opacity = "0"; await new Promise(r => setTimeout(r, 300));

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
        const printBtn = document.getElementById("printBtn");
        if (!printBtn.onclick) {
          printBtn.onclick = () => printResult(jobId, printBtn);
          printBtn.disabled = false;
        }
        const [past, future10, future20] = s.target_ages;
        document.getElementById("ageNote").textContent =
          `${past}세 편집 → 현재 원본(약 ${s.initial_age}세) → ${future10}세 편집 → ${future20}세 편집`;
        show("done");
      } else if (s.status === "error") {
        clearInterval(poll); clearInterval(timer);
        fail(s.error || "생성에 실패했어요.");
      }
    } catch (e) { /* 일시적 네트워크 오류는 무시하고 계속 폴링 */ }
  }, 2000);
}

async function printResult(jobId, button) {
  if (button.disabled) return;
  button.disabled = true;
  const msg = document.getElementById("printMsg");
  msg.textContent = "프린터로 보내는 중이에요…";
  try {
    const response = await fetch("/print_result/" + jobId, {method: "POST"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    msg.textContent = "프린터로 보냈어요.";
  } catch (e) {
    msg.textContent = e.message || "인쇄 상태를 확인해 주세요.";
  } finally {
    button.disabled = false;
  }
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


def _grab(warmup=2, tries=15, timeout=3.0):
    """해상도 변경 직후의 빈/묵은 프레임 몇 장만 버리고, 그다음 첫 정상 프레임을 바로 쓴다.
    (여러 장을 다 읽으면 고해상도에서 느린 웹캠은 촬영이 몇 초씩 늦어진다)"""
    skipped, deadline = 0, time.monotonic() + timeout
    for _ in range(tries):
        if time.monotonic() > deadline:
            break
        success, f = read_camera_frame()
        if success and f is not None:
            if skipped < warmup:
                skipped += 1
                continue
            return f
        time.sleep(0.03)
    return None


def capture_frame():
    with camera_lock:
        try:
            started = time.monotonic()
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            frame = _grab()
            if frame is None:
                # 고해상도를 지원하지 않는 웹캠이면 미리보기 해상도로 다시 촬영
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                frame = _grab()
            app.logger.warning("capture took %.2fs", time.monotonic() - started)
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

        # A relay or login page may return HTML with status 200.
        try:
            with Image.open(io.BytesIO(resp.content)) as result_image:
                if result_image.format != "PNG":
                    raise ValueError("Expected PNG")
                result_image.verify()
        except (OSError, ValueError, SyntaxError):
            _set_job(job_id, status="error", error="서버에서 온 네 컷 이미지를 읽지 못했어요. 다시 시도해 주세요.")
            return
        # Never label a legacy fixed-target response as a relative-age sheet.
        try:
            initial_age = int(resp.headers["X-Initial-Age"])
            targets = [int(value) for value in resp.headers["X-Target-Ages"].split(",")]
            if len(targets) != 3 or not all(3 <= age <= 90 for age in targets):
                raise ValueError("Invalid targets")
        except (KeyError, ValueError):
            _set_job(job_id, status="error", error="결과의 편집 나이를 확인할 수 없어요. 서버 설정을 확인해 주세요.")
            return
        (RESULT_DIR / f"{job_id}.png").write_bytes(resp.content)

        # 결과 다운로드 URL로 QR 생성 (핸드폰이 같은 네트워크에서 접속)
        url = f"http://{PUBLIC_HOST}:{WEB_PORT}/result/{job_id}?download=1"
        qrcode.make(url, box_size=QR_BOX_SIZE).save(QR_DIR / f"{job_id}.png")

        age = resp.headers.get("X-Estimated-Age", "")
        _set_job(job_id, status="done", age=age, initial_age=initial_age, target_ages=targets,
                 gpu_job_id=resp.headers.get("X-Job-Id", ""), url=url)
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
                   age=job.get("age", ""), initial_age=job.get("initial_age"),
                   target_ages=job.get("target_ages", []), gpu_job_id=job.get("gpu_job_id", ""),
                   error=job.get("error", ""))


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


def process_and_print_image(pil_img, caption="", qr_img=None):
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
    # Scale the saved QR in whole printer dots per module, including its quiet zone.
    if qr_img is not None:
        modules = qr_img.width // QR_BOX_SIZE
        if not 0 < modules <= PRINTER_WIDTH or qr_img.size != (modules * QR_BOX_SIZE,) * 2:
            raise ValueError("Unexpected saved QR dimensions")
        size = modules * (PRINTER_WIDTH // modules)
        qr_img = qr_img.convert("1").resize((size, size), Image.Resampling.NEAREST)
    final_img = Image.new("1", (PRINTER_WIDTH, header_h + img.height + footer_h +
                                (qr_img.height if qr_img is not None else 0)), 1)
    draw = ImageDraw.Draw(final_img)
    draw.text((20, 12), "AI TIME MACHINE PHOTO", fill=0, font=font)
    draw.line([(0, header_h - 5), (PRINTER_WIDTH, header_h - 5)], fill=0, width=2)
    final_img.paste(img, (0, header_h))
    footer_top = header_h + img.height
    draw.line([(0, footer_top + 5), (PRINTER_WIDTH, footer_top + 5)], fill=0, width=2)
    if caption:
        draw.text((20, footer_top + 15), caption, fill=0, font=font)
    if qr_img is not None:
        final_img.paste(qr_img, ((PRINTER_WIDTH - qr_img.width) // 2, footer_top + footer_h))

    byte_w = PRINTER_WIDTH // 8
    header = bytearray([0x1D, 0x76, 0x30, 0x00, byte_w & 0xFF, (byte_w >> 8) & 0xFF,
                        final_img.height & 0xFF, (final_img.height >> 8) & 0xFF])
    # ESC/POS uses a set bit for black; Pillow mode 1 uses it for white.
    payload = header + final_img.tobytes("raw", "1;I")
    with serial.Serial("/dev/ttyACM0", 115200, timeout=2, write_timeout=5) as ser:
        wake_up_and_clear_printer(ser)
        for i in range(0, len(payload), 1024):
            ser.write(payload[i:i+1024]); time.sleep(0.01)
        ser.write(b"\n\n\n\n"); ser.flush()


@app.post("/print_result/<job_id>")
def print_result(job_id):
    with jobs_lock:
        state = jobs.get(job_id, {}).get("status") if ID_RE.fullmatch(job_id) else None
    if state is None:
        return jsonify(error="작업을 찾을 수 없어요."), 404
    if state != "done":
        return jsonify(error="완료된 결과만 인쇄할 수 있어요."), 409
    if not printer_busy.acquire(blocking=False):
        return jsonify(error="프린터 사용 중이에요. 잠시 후 다시 시도해 주세요."), 409
    try:
        try:
            with Image.open(RESULT_DIR / f"{job_id}.png") as image:
                photo = image.copy()
            with Image.open(QR_DIR / f"{job_id}.png") as image:
                qr = image.copy()
        except FileNotFoundError:
            return jsonify(error="저장된 결과 또는 QR이 없어요."), 404
        process_and_print_image(photo, qr_img=qr)
        return jsonify(job_id=job_id)
    except Exception:
        app.logger.exception("Result printing failed for %s", job_id)
        return jsonify(error="인쇄에 실패했어요. 저장된 결과와 프린터를 확인해 주세요."), 500
    finally:
        printer_busy.release()


@app.route("/capture_and_print", methods=["POST"])
def capture_and_print():
    if not capture_busy.acquire(blocking=False):
        return render_template_string(
            HTML_TEMPLATE, status="다른 촬영이 진행 중이에요. 잠시 후 다시 시도해 주세요."), 409
    try:
        if not printer_busy.acquire(blocking=False):
            return render_template_string(HTML_TEMPLATE, status="프린터 사용 중이에요."), 409
        try:
            caption = request.form.get("caption", "").strip()
            frame = capture_frame()
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            process_and_print_image(pil_img, caption)
        finally:
            printer_busy.release()
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
