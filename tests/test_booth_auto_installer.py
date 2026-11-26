"""Hardware-free checks; generated images are synthetic, never user photos."""
import importlib.util
import io
import json
from pathlib import Path
import threading
import time
import types

import pytest
from flask import Flask, abort, render_template_string, request
from PIL import Image


spec = importlib.util.spec_from_file_location(
    "booth_installer", Path(__file__).resolve().parents[1] / "tools/install_booth_auto.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


FIXTURE = '''# booth-button-capture-v1
from flask import Flask, request, render_template_string
app = Flask(__name__)
camera_lock = object()
capture_busy = object()
HTML_TEMPLATE = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><style>body{margin:20px}</style></head><body><div class="card"><h2>AI 타임머신 포토부스</h2><div class="webcam-container">카메라 미리보기 위치</div><form method="POST" action="/capture_only" id="capture-form"><button>촬영하고 저장</button></form><form><button>촬영하고 인쇄하기</button></form></div></body></html>"""
def generate_frames():
    pass
def capture_frame():
    pass
@app.route("/capture_only", methods=["POST"])
def capture_only():
    pass
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''


class InlineThread:
    def __init__(self, target, args, daemon):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


@pytest.fixture
def app_namespace(tmp_path, monkeypatch):
    app = Flask(__name__)
    app.config["TESTING"] = True

    def encode(ext, frame, options):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "#789abc").save(buffer, "JPEG")
        return True, types.SimpleNamespace(tobytes=buffer.getvalue)

    namespace = {
        "app": app, "re": __import__("re"), "time": time,
        "request": request, "abort": abort, "Image": Image,
        "capture_frame": lambda: object(), "capture_busy": threading.Lock(),
        "cv2": types.SimpleNamespace(imencode=encode, IMWRITE_JPEG_QUALITY=1),
    }
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    exec(installer.EXTENSION, namespace)
    namespace["AUTO_ROOT"] = tmp_path / "jobs"
    namespace["AUTO_ROOT"].mkdir()
    namespace["_auto_base_url"] = lambda: "http://192.168.1.50:5000"
    namespace["_auto_threading"] = types.SimpleNamespace(Thread=InlineThread)
    return app, namespace


class FakeResponse:
    def __init__(self, status=200, content=None):
        self.status_code = status
        self.headers = {"X-Estimated-Age": "21.2"}
        if content is None:
            buffer = io.BytesIO()
            Image.new("RGB", (256, 256), "#aabbcc").save(buffer, "PNG")
            content = buffer.getvalue()
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.content


def start(client):
    return client.post("/api/booth/jobs", json={"gender": "female", "consent": True})


def test_installer_preserves_code_and_refuses_repeat(tmp_path):
    updated = installer.build_source(FIXTURE)
    compile(updated, "app.py", "exec")
    assert "def capture_frame():" in updated
    assert '<form method="POST" action="/capture_only" id="capture-form">' in updated
    assert "촬영하고 인쇄하기" in updated
    assert "auto_booth_start" in updated
    with pytest.raises(ValueError, match="이미 설치"):
        installer.build_source(updated)
    path = tmp_path / "app.py"
    path.write_text(FIXTURE)
    backup = installer.install(path)
    assert backup.read_text() == FIXTURE
    assert installer.MARKER in path.read_text()


def test_invalid_input_does_not_capture(app_namespace):
    app, ns = app_namespace
    ns["capture_frame"] = lambda: pytest.fail("must not capture")
    client = app.test_client()
    assert client.post("/api/booth/jobs", json={}).status_code == 400
    assert client.post("/api/booth/jobs", json={"gender": "female"}).status_code == 400
    ns["capture_busy"].acquire()
    assert start(client).status_code == 409
    ns["capture_busy"].release()


def test_success_result_qr_and_expiry(app_namespace, monkeypatch):
    app, ns = app_namespace
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(ns["_auto_requests"], "post", post)
    client = app.test_client()
    response = start(client)
    assert response.status_code == 202
    job_id = response.json["job_id"]
    status = client.get("/api/booth/jobs/" + job_id)
    assert status.json["status"] == "done"
    assert status.json["estimated_age"] == 21.2
    assert calls[0][0].endswith("/api/fading/fourcut")
    assert calls[0][1]["data"] == {"gender": "female"}
    assert client.get(status.json["image_url"]).content_type == "image/png"
    assert "attachment" in client.get(status.json["download_url"]).headers["Content-Disposition"]
    assert client.get(status.json["qr_url"]).content_type == "image/png"
    state = ns["_auto_read"](job_id)
    assert state["public_download_url"] == f"http://192.168.1.50:5000/auto-results/{job_id}.png?download=1"
    state["expires_at"] = time.time() - 1
    ns["_auto_write"](job_id, state)
    assert client.get(status.json["download_url"]).status_code == 410
    assert client.get(status.json["qr_url"]).status_code == 410
    assert client.get("/api/booth/jobs/" + job_id).status_code == 410
    assert (ns["AUTO_ROOT"] / job_id / "fourcut.png").exists()
    assert not ns["capture_busy"].locked()


@pytest.mark.parametrize("status", [409, 422, 500, 504, 507])
def test_backend_error_never_publishes_photo(app_namespace, monkeypatch, status):
    app, ns = app_namespace
    monkeypatch.setattr(ns["_auto_requests"], "post", lambda *a, **k: FakeResponse(status=status))
    client = app.test_client()
    job = start(client).json["job_id"]
    assert client.get("/api/booth/jobs/" + job).json["status"] == "error"
    assert client.get("/auto-results/" + job + ".png").status_code == 409
    assert not ns["capture_busy"].locked()


def test_bad_png_is_not_served(app_namespace, monkeypatch):
    app, ns = app_namespace
    monkeypatch.setattr(ns["_auto_requests"], "post", lambda *a, **k: FakeResponse(content=b"not a photo"))
    client = app.test_client()
    job = start(client).json["job_id"]
    assert client.get("/api/booth/jobs/" + job).json["status"] == "error"
    assert client.get("/auto-results/" + job + ".png").status_code == 409


def test_network_failure_unlocks_capture(app_namespace, monkeypatch):
    app, ns = app_namespace

    def fail(*args, **kwargs):
        raise ns["_auto_requests"].ConnectionError("test only")

    monkeypatch.setattr(ns["_auto_requests"], "post", fail)
    client = app.test_client()
    job = start(client).json["job_id"]
    assert client.get("/api/booth/jobs/" + job).json["status"] == "error"
    assert not ns["capture_busy"].locked()


def test_restart_marks_interrupted_job_as_error(app_namespace):
    app, ns = app_namespace
    job = "a" * 32
    (ns["AUTO_ROOT"] / job).mkdir()
    ns["_auto_write"](job, {
        "status": "processing", "message": "old", "created": time.time(),
        "run_id": "old-run",
    })
    assert app.test_client().get("/api/booth/jobs/" + job).json["status"] == "error"
    assert app.test_client().get("/auto-results/not-a-job.png").status_code == 404


def test_health_checks_real_backend(app_namespace, monkeypatch):
    app, ns = app_namespace
    monkeypatch.setattr(ns["_auto_requests"], "get", lambda *a, **k: types.SimpleNamespace(
        status_code=200,
        json=lambda: {"service": "photobooth-fading", "real_age_model": True},
    ))
    assert app.test_client().get("/api/booth/health").json["hai_connected"] is True
    monkeypatch.setattr(ns["_auto_requests"], "get", lambda *a, **k: types.SimpleNamespace(
        status_code=200, json=lambda: {"service": "photobooth-connection-test"},
    ))
    assert app.test_client().get("/api/booth/health").status_code == 503


def test_page_fixture_can_render(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    namespace = {"__name__": "test_fixture"}
    exec(installer.build_source(FIXTURE), namespace)
    with namespace["app"].test_request_context("/"):
        html = render_template_string(namespace["HTML_TEMPLATE"])
    (tmp_path / "booth.html").write_text(html)
    assert "auto-form" in html
    assert "auto-qr" in html
