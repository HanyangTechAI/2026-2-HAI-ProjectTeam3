#!/usr/bin/env python3
"""Add LAN-only automatic FADING/QR controls to the existing Raspberry Pi app.

Run with the Pi app's Python after installing requests and qrcode there.
Does not start/stop services, change the HAI program, or delete photographs.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import shutil

MARKER = "# photobooth-auto-qr-v1"

STYLE = """
        body { font-family: Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", system-ui, sans-serif; word-break: keep-all; line-height: 1.75; }
        .auto-panel { text-align: left; border-top: 1px solid #c8d0dc; padding-top: 18px; margin-top: 18px; }
        .auto-panel h3 { letter-spacing: -.02em; margin: 0 0 12px; }
        .auto-panel label { display: block; margin: 12px 0 6px; }
        .auto-panel select { width: 100%; min-height: 44px; font: inherit; padding: 8px; border: 1px solid #65758b; border-radius: 8px; background: white; color: #1c1c1e; }
        .auto-panel .auto-consent { display: flex; align-items: flex-start; gap: 9px; font-size: 14px; }
        .auto-consent input { flex-shrink: 0; width: 20px; height: 20px; margin-top: 4px; }
        .auto-panel button { background: #005bc4; min-height: 48px; margin-top: 10px; }
        .auto-panel button:hover:not(:disabled) { background: #004695; }
        button, a, select, input { touch-action: manipulation; }
        button:disabled { background: #52647b; cursor: wait; }
        button:focus-visible, a:focus-visible, select:focus-visible, input:focus-visible { outline: 3px solid #12324e; outline-offset: 3px; }
        .auto-note { color: #3e4b5b; font-size: 14px; }
        #auto-error { color: #a51e24; }
        #auto-result img { display: block; max-width: 100%; height: auto; margin: 12px auto; }
        #auto-qr { width: 240px; background: white; }
        #auto-result a { display: inline-block; min-height: 44px; color: #0053ad; font-weight: 600; padding: 8px 0; }
        [hidden] { display: none !important; }
"""

PANEL = """
        <section class="auto-panel" aria-labelledby="auto-heading">
            <h3 id="auto-heading">AI 연령변환 4컷</h3>
            <p class="auto-note">원본 · 목표 10세 · 30세 · 50세를 만들어요. 실제 과거·미래 얼굴을 예측하는 것은 아니에요.</p>
            <form id="auto-form" autocomplete="off">
                <label for="auto-gender">연령변환 성별 조건 *</label>
                <select id="auto-gender" name="gender" required>
                    <option value="">선택하세요</option>
                    <option value="female">여성</option>
                    <option value="male">남성</option>
                </select>
                <label class="auto-consent" for="auto-consent">
                    <input id="auto-consent" name="consent" type="checkbox" required>
                    <span>촬영 사진을 HAI로 보내 연령변환하고, 사진·결과를 HAI와 라즈베리파이에 저장하는 데 동의해요. 파일은 자동 삭제되지 않아요.</span>
                </label>
                <button type="submit">촬영하고 AI 4컷 만들기</button>
            </form>
            <p id="auto-status" role="status" aria-live="polite">버튼을 누르면 3초 뒤 촬영해요. 생성에는 약 2분 이상 걸릴 수 있어요.</p>
            <p id="auto-error" role="alert" hidden></p>
            <button id="auto-retry" type="button" hidden>처리 상태 다시 확인</button>
            <section id="auto-result" aria-labelledby="auto-result-heading" hidden>
                <h3 id="auto-result-heading">완성된 4컷</h3>
                <p id="auto-metadata" class="auto-note"></p>
                <img id="auto-sheet" width="1024" height="1112" loading="lazy" alt="촬영 원본과 목표 연령별 변환 결과 4컷">
                <a id="auto-download">4컷 다운로드</a>
                <h3>휴대전화로 받기</h3>
                <img id="auto-qr" width="240" height="240" loading="lazy" alt="완성된 4컷 다운로드 QR 코드">
                <p class="auto-note">라즈베리파이와 같은 와이파이에서 스캔해 주세요. 다운로드 링크는 완성 후 1시간 동안 열려요. 인쇄는 자동으로 하지 않아요.</p>
            </section>
        </section>
        <p class="auto-note">아래는 기존 원본 촬영·인쇄 기능이에요.</p>
"""

JAVASCRIPT = r"""
    <script>
    (() => {
        const form = document.getElementById('auto-form');
        const status = document.getElementById('auto-status');
        const error = document.getElementById('auto-error');
        const retry = document.getElementById('auto-retry');
        const result = document.getElementById('auto-result');
        const key = 'photobooth-auto-current-job-v1';
        let running = false;
        const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
        function storedJob() {
            try { return sessionStorage.getItem(key); } catch (_) { return null; }
        }
        function storeJob(id) {
            try { id ? sessionStorage.setItem(key, id) : sessionStorage.removeItem(key); } catch (_) {}
        }
        function busy(value) {
            running = value;
            document.querySelectorAll('button, select, input[type="checkbox"]').forEach(el => {
                if (el !== retry) el.disabled = value;
            });
        }
        async function json(url, options = {}) {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 15000);
            try {
                const response = await fetch(url, {...options, signal: controller.signal, cache: 'no-store'});
                let data;
                try { data = await response.json(); } catch (_) {
                    throw new Error('응답을 읽지 못했어요. 연결 상태를 확인해 주세요.');
                }
                if (!response.ok) {
                    const failure = new Error(data.error || '요청을 처리하지 못했어요.');
                    failure.status = response.status;
                    throw failure;
                }
                return data;
            } finally { clearTimeout(timer); }
        }
        async function poll(id) {
            const began = Date.now();
            while (Date.now() - began < 12 * 60 * 1000) {
                const job = await json('/api/booth/jobs/' + encodeURIComponent(id));
                if (job.status === 'error') {
                    storeJob(null);
                    throw new Error(job.message);
                }
                const elapsed = Math.max(0, Math.floor(Date.now() / 1000 - job.created));
                status.textContent = job.message + ' · ' + elapsed + '초 경과';
                if (job.status === 'done') {
                    document.getElementById('auto-sheet').src = job.image_url;
                    document.getElementById('auto-qr').src = job.qr_url;
                    document.getElementById('auto-download').href = job.download_url;
                    const age = job.estimated_age == null ? '확인되지 않음' : job.estimated_age.toFixed(1) + '세';
                    document.getElementById('auto-metadata').textContent =
                        '추정 나이: ' + age + ' · 촬영 요청부터 QR 완성까지 ' + job.total_seconds.toFixed(1) + '초';
                    result.hidden = false;
                    status.textContent = '4컷과 QR이 완성됐어요. 아래에서 확인해 주세요.';
                    document.getElementById('auto-gender').value = '';
                    document.getElementById('auto-consent').checked = false;
                    retry.hidden = true;
                    busy(false);
                    return;
                }
                await sleep(2000);
            }
            throw new Error('처리가 오래 걸리고 있어요. 새 촬영 대신 처리 상태를 다시 확인해 주세요.');
        }
        function failed(failure) {
            if (failure.status === 404 || failure.status === 410) storeJob(null);
            error.textContent = failure.name === 'AbortError'
                ? '연결 응답이 늦어요. 새 촬영 대신 처리 상태를 다시 확인해 주세요.'
                : failure.message;
            error.hidden = false;
            const pending = Boolean(storedJob());
            retry.hidden = !pending;
            busy(pending);
        }
        form.addEventListener('submit', async event => {
            event.preventDefault();
            if (running || !form.reportValidity()) return;
            const gender = document.getElementById('auto-gender').value;
            error.hidden = true;
            retry.hidden = true;
            result.hidden = true;
            storeJob(null);
            busy(true);
            try {
                for (let count = 3; count >= 1; count--) {
                    status.textContent = count + '초 뒤 촬영해요. 정면을 바라봐 주세요.';
                    await sleep(1000);
                }
                status.textContent = '촬영 요청을 보내고 있어요.';
                const job = await json('/api/booth/jobs', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({gender, consent: true}),
                });
                storeJob(job.job_id);
                await poll(job.job_id);
            } catch (failure) { failed(failure); }
        });
        async function resume() {
            const id = storedJob();
            if (!id) return;
            busy(true);
            retry.hidden = true;
            error.hidden = true;
            try { await poll(id); } catch (failure) { failed(failure); }
        }
        retry.addEventListener('click', resume);
        resume();
    })();
    </script>
"""

EXTENSION = r'''
# photobooth-auto-qr-v1
import json as _auto_json
import math as _auto_math
import os as _auto_os
import socket as _auto_socket
import threading as _auto_threading
import uuid as _auto_uuid
from pathlib import Path as _AutoPath
from urllib.parse import urlsplit as _auto_urlsplit
import requests as _auto_requests
import qrcode as _auto_qrcode
from flask import jsonify as _auto_jsonify, send_file as _auto_send_file

AUTO_ROOT = _AutoPath.home() / "Pictures/ai_booth/auto"
AUTO_ROOT.mkdir(parents=True, exist_ok=True)
AUTO_RUN_ID = _auto_uuid.uuid4().hex
AUTO_HAI_URL = "http://127.0.0.1:18000"
AUTO_LINK_SECONDS = 3600


def _auto_reply(data, status=200):
    response = _auto_jsonify(data)
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    return response


def _auto_id_valid(job_id):
    return bool(re.fullmatch(r"[a-f0-9]{32}", job_id))


def _auto_write(job_id, data):
    folder = AUTO_ROOT / job_id
    temporary = folder / ("status_" + _auto_uuid.uuid4().hex + ".tmp")
    temporary.write_text(_auto_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(folder / "status.json")


def _auto_read(job_id):
    if not _auto_id_valid(job_id):
        return None
    try:
        return _auto_json.loads((AUTO_ROOT / job_id / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _auto_base_url():
    configured = _auto_os.environ.get("BOOTH_PUBLIC_BASE_URL", "").rstrip("/")
    if configured:
        parsed = _auto_urlsplit(configured)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("Invalid BOOTH_PUBLIC_BASE_URL")
        return configured
    # UDP connect only selects a local interface; it does not send a packet.
    with _auto_socket.socket(_auto_socket.AF_INET, _auto_socket.SOCK_DGRAM) as sock:
        sock.connect(("192.0.2.1", 9))
        address = sock.getsockname()[0]
    if address.startswith("127.") or address == "0.0.0.0":
        raise ValueError("No usable LAN address")
    return "http://" + address + ":5000"


def _auto_number(value):
    try:
        number = float(value)
        return number if _auto_math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _auto_worker(job_id, gender, state):
    folder = AUTO_ROOT / job_id
    try:
        state["message"] = "사진을 촬영하고 있어요"
        _auto_write(job_id, state)
        frame = capture_frame()
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("카메라 사진을 저장하지 못했어요.")
        photo = encoded.tobytes()
        if len(photo) > 10 * 1024 * 1024:
            raise RuntimeError("촬영 사진이 10MB를 넘어요.")
        with (folder / "input.jpg").open("xb") as file:
            file.write(photo)

        state["message"] = "HAI에서 나이 추정과 연령변환을 진행하고 있어요"
        _auto_write(job_id, state)
        partial = folder / "fourcut.partial"
        with _auto_requests.post(
            AUTO_HAI_URL + "/api/fading/fourcut",
            files={"file": ("capture.jpg", photo, "image/jpeg")},
            data={"gender": gender},
            timeout=(10, 660),
            stream=True,
        ) as response:
            if response.status_code != 200:
                messages = {
                    409: "HAI가 다른 사진을 처리 중이에요. 완료 후 다시 촬영해 주세요.",
                    422: "얼굴 한 명이 선명하게 나오도록 다시 촬영해 주세요.",
                    504: "HAI 생성 시간이 초과됐어요. 실행 기록을 확인해 주세요.",
                    507: "HAI 저장 공간이 부족해요.",
                }
                raise RuntimeError(messages.get(response.status_code,
                    "HAI에서 4컷을 만들지 못했어요. HAI 실행 기록을 확인해 주세요."))
            received = 0
            with partial.open("xb") as file:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > 20 * 1024 * 1024:
                        raise RuntimeError("생성 결과가 예상 크기를 넘어요.")
                    file.write(chunk)
            estimated = _auto_number(response.headers.get("X-Estimated-Age"))

        with Image.open(partial) as image:
            if image.format != "PNG" or image.width * image.height > 20_000_000:
                raise RuntimeError("HAI 결과가 올바른 사진이 아니에요.")
            image.verify()
        partial.replace(folder / "fourcut.png")

        state["message"] = "완성된 사진의 다운로드 QR을 만들고 있어요"
        _auto_write(job_id, state)
        download = state["public_base"] + "/auto-results/" + job_id + ".png?download=1"
        qr = _auto_qrcode.QRCode(
            error_correction=_auto_qrcode.constants.ERROR_CORRECT_M,
            box_size=8, border=4,
        )
        qr.add_data(download)
        qr.make(fit=True)
        qr.make_image(fill_color="black", back_color="white").save(folder / "qr.png")
        finished = time.time()
        state.update({
            "status": "done", "message": "4컷과 QR이 완성됐어요",
            "estimated_age": estimated,
            "total_seconds": finished - state["created"],
            "expires_at": finished + AUTO_LINK_SECONDS,
            "public_download_url": download,
        })
        _auto_write(job_id, state)
    except _auto_requests.Timeout:
        app.logger.exception("Automatic booth request timed out: %s", job_id)
        state.update(status="error", message="HAI 응답 시간이 초과됐어요. 다시 촬영하기 전에 HAI 처리 상태를 확인해 주세요.")
        _auto_write(job_id, state)
    except _auto_requests.ConnectionError:
        app.logger.exception("Automatic booth connection failed: %s", job_id)
        state.update(status="error", message="HAI에 연결하지 못했어요. 맥의 중계 연결과 HAI 실행 상태를 확인해 주세요.")
        _auto_write(job_id, state)
    except Exception as error:
        app.logger.exception("Automatic booth failed: %s", job_id)
        message = str(error) if isinstance(error, RuntimeError) else "촬영 또는 결과 저장에 실패했어요. 라즈베리파이 실행 기록을 확인해 주세요."
        state.update(status="error", message=message)
        _auto_write(job_id, state)
    finally:
        capture_busy.release()


@app.route("/api/booth/health")
def auto_booth_health():
    try:
        response = _auto_requests.get(AUTO_HAI_URL + "/health", timeout=(3, 5))
        data = response.json()
        if response.status_code != 200 or data.get("service") != "photobooth-fading" or not data.get("real_age_model"):
            raise ValueError("Unexpected service")
        return _auto_reply({"status": "ok", "hai_connected": True, "qr_base": _auto_base_url()})
    except Exception:
        return _auto_reply({"error": "HAI 연결 또는 QR 다운로드 주소를 확인하지 못했어요."}, 503)


@app.route("/api/booth/jobs", methods=["POST"])
def auto_booth_start():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("gender") not in ("female", "male"):
        return _auto_reply({"error": "연령변환 성별 조건을 선택해 주세요."}, 400)
    if data.get("consent") is not True:
        return _auto_reply({"error": "사진 전송·저장 안내에 동의해 주세요."}, 400)
    if not capture_busy.acquire(blocking=False):
        return _auto_reply({"error": "다른 촬영·생성이 진행 중이에요. 완료 후 다시 시도해 주세요."}, 409)
    job_id = _auto_uuid.uuid4().hex
    folder = AUTO_ROOT / job_id
    try:
        public_base = _auto_base_url()
        folder.mkdir(exist_ok=False)
        state = {
            "status": "processing", "message": "촬영을 준비하고 있어요",
            "created": time.time(), "run_id": AUTO_RUN_ID,
            "public_base": public_base,
        }
        _auto_write(job_id, state)
        worker = _auto_threading.Thread(
            target=_auto_worker, args=(job_id, data["gender"], state), daemon=True,
        )
        worker.start()
        return _auto_reply({"job_id": job_id}, 202)
    except Exception:
        capture_busy.release()
        app.logger.exception("Could not start automatic booth job")
        return _auto_reply({"error": "촬영을 시작하지 못했어요. 저장 공간과 네트워크를 확인해 주세요."}, 500)


@app.route("/api/booth/jobs/<job_id>")
def auto_booth_status(job_id):
    state = _auto_read(job_id)
    if state is None:
        return _auto_reply({"error": "촬영 기록을 찾지 못했어요."}, 404)
    if state["status"] == "done" and time.time() > state["expires_at"]:
        return _auto_reply({"error": "다운로드 링크의 1시간 이용 기간이 지났어요."}, 410)
    if state["status"] == "processing" and state.get("run_id") != AUTO_RUN_ID:
        state.update(status="error", message="포토부스가 재시작돼 처리 결과를 확인하지 못했어요. HAI 상태를 확인해 주세요.")
    result = {key: state[key] for key in ("status", "message", "created")}
    if state["status"] == "done":
        result.update(
            estimated_age=state.get("estimated_age"),
            total_seconds=state["total_seconds"],
            image_url="/auto-results/" + job_id + ".png",
            download_url="/auto-results/" + job_id + ".png?download=1",
            qr_url="/auto-qr/" + job_id + ".png",
        )
    return _auto_reply(result)


def _auto_file(job_id, filename, attachment=False):
    state = _auto_read(job_id)
    if state is None:
        abort(404)
    if state["status"] != "done":
        abort(409)
    if time.time() > state["expires_at"]:
        abort(410)
    path = AUTO_ROOT / job_id / filename
    if not path.is_file():
        abort(404)
    response = _auto_send_file(
        path, mimetype="image/png", as_attachment=attachment,
        download_name="ai_fourcut_" + job_id + ".png" if attachment else None,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/auto-results/<job_id>.png")
def auto_booth_image(job_id):
    return _auto_file(job_id, "fourcut.png", request.args.get("download") == "1")


@app.route("/auto-qr/<job_id>.png")
def auto_booth_qr(job_id):
    return _auto_file(job_id, "qr.png")
'''


def build_source(source):
    """Validate the known existing app, then add routes without importing it."""
    if MARKER in source:
        raise ValueError("자동 4컷·QR 기능이 이미 설치돼 있어요. 중복 적용하지 않았어요.")
    if "# booth-button-capture-v1" not in source:
        raise ValueError("앞서 추가한 촬영·저장 버튼이 있는 app.py가 필요해요.")
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not {"capture_frame", "generate_frames", "capture_only", "index"} <= names:
        raise ValueError("기존 촬영 함수 구성이 예상과 달라 중단했어요.")
    for name in ("capture_busy", "camera_lock"):
        if name not in source:
            raise ValueError("카메라 잠금 구성이 없어서 중단했어요.")
    templates = [
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "HTML_TEMPLATE" for target in node.targets)
    ]
    if len(templates) != 1:
        raise ValueError("화면 템플릿 구성을 확인하지 못했어요.")
    node = templates[0]
    html = ast.literal_eval(node.value)
    html = html.replace("<html>", '<html lang="ko">', 1)
    anchor = '<form method="POST" action="/capture_only" id="capture-form">'
    if html.count(anchor) != 1 or html.count("</body>") != 1 or html.count("</style>") != 1:
        raise ValueError("화면 수정 위치가 예상과 달라 중단했어요.")
    html = html.replace(anchor, PANEL + "\n" + anchor, 1)
    html = html.replace("</style>", STYLE + "\n</style>", 1)
    html = html.replace("</body>", JAVASCRIPT + "\n</body>", 1)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ["HTML_TEMPLATE = " + repr(html) + "\n"]
    source = "".join(lines)
    tree = ast.parse(source)
    mains = [
        node for node in tree.body if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
        and any(isinstance(item, ast.Constant) and item.value == "__main__" for item in node.test.comparators)
    ]
    if len(mains) != 1:
        raise ValueError("기존 실행 구성을 확인하지 못했어요.")
    lines = source.splitlines(keepends=True)
    lines[mains[0].lineno - 1:mains[0].lineno - 1] = ["\n" + EXTENSION + "\n"]
    updated = "".join(lines)
    compile(updated, "photobooth/app.py", "exec")
    return updated


def install(path):
    source = path.read_text(encoding="utf-8")
    updated = build_source(source)
    # Check dependencies in the actual Python used to run this installer.
    import requests  # noqa: F401
    import qrcode  # noqa: F401

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name("app.py.before_auto_" + stamp)
    shutil.copy2(path, backup)
    temporary = path.with_name("app.py.auto_tmp_" + stamp)
    temporary.write_text(updated, encoding="utf-8")
    temporary.chmod(path.stat().st_mode & 0o777)
    temporary.replace(path)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=Path("/home/photobooth/photobooth/app.py"))
    args = parser.parse_args()
    try:
        backup = install(args.app)
    except (OSError, ValueError, ImportError, SyntaxError) as error:
        raise SystemExit("설치 중단: " + str(error))
    print("자동 촬영·4컷·QR 코드 설치 및 문법 확인 완료")
    print("기존 파일 백업:", backup)
    print("아직 실행 중인 프로그램에는 적용되지 않았어요.")
    print("다음: sudo systemctl restart photobooth.service")
    print("기존 인쇄 기능은 유지되며, AI 결과는 자동 인쇄하지 않아요.")


if __name__ == "__main__":
    main()
