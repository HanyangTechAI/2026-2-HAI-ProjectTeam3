import importlib.util
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest
import qrcode
import requests
from PIL import Image
from werkzeug.serving import make_server


JOB = "a" * 32
OTHER = "b" * 32


@pytest.fixture
def capture_booth(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PUBLIC_HOST", "127.0.0.1")
    monkeypatch.setattr(cv2, "VideoCapture", Mock())
    path = Path(__file__).resolve().parents[1] / "raspi/photobooth_app.py"
    spec = importlib.util.spec_from_file_location("test_booth_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.serial, "Serial", Mock(side_effect=AssertionError("unexpected print")))
    return module


@pytest.fixture
def booth(capture_booth):
    capture_booth.capture_frame = Mock(side_effect=AssertionError("unexpected capture"))
    yield capture_booth
    capture_booth.capture_frame.assert_not_called()


def seed(booth, job, base, color=255):
    # Four distinct quadrants, aligned to the native printer width.
    image = np.zeros((128, 384), dtype=np.uint8)
    image[:64, :192] = image[64:, 192:] = color
    Image.fromarray(image).save(booth.RESULT_DIR / f"{job}.png")
    url = f"{base}/result/{job}?download=1"
    qrcode.make(url).save(booth.QR_DIR / f"{job}.png")
    booth._set_job(job, status="done", url=url)
    return url


def decode(image):
    return cv2.QRCodeDetector().detectAndDecode(np.array(image.convert("L")))[0]


def test_mock_capture_to_color_download(capture_booth, monkeypatch):
    # Fixed URL makes decoder coverage reproducible; no host/network dependencies.
    booth = capture_booth
    booth.capture_frame = Mock(wraps=booth.capture_frame)
    monkeypatch.setattr(booth, 'uuid', SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=JOB)))
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    frame[:, :32] = (255, 0, 0)
    booth.camera.read.return_value = (True, frame)
    # Distinct marks in each of the four slots survive monochrome printing.
    dark = np.zeros((128, 384), dtype=bool)
    for slot in range(4):
        y, x = (slot // 2) * 64, (slot % 2) * 192
        dark[y+8:y+56, x+16:x+32+slot*24] = True
    pixels = np.where(dark[:, :, None], [0, 1, 2], [255, 254, 253]).astype('uint8')
    png = io.BytesIO()
    Image.fromarray(pixels).save(png, format='PNG')
    response = requests.Response()
    response.status_code = 200
    response._content = png.getvalue()
    response.headers.update({'X-Estimated-Age': '84.8', 'X-Initial-Age': '85',
                             'X-Target-Ages': '75,90,90', 'X-Job-Id': OTHER})

    def gpu(url, files, data, timeout):
        assert url == booth.FADING_API_URL and data == {'gender': 'female'}
        assert files['file'][1].read() == cv2.imencode('.jpg', cv2.flip(frame, 1),
                                                       [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        return response

    post = Mock(side_effect=gpu)
    monkeypatch.setattr(booth.requests, 'post', post)
    finished = threading.Event()
    worker = booth.run_fading_job

    def observe(*args):
        try:
            worker(*args)
        finally:
            finished.set()

    monkeypatch.setattr(booth, 'run_fading_job', observe)
    printer = io.BytesIO()
    write = Mock(wraps=printer.write)
    printer.write = write
    booth.serial.Serial.side_effect = None
    booth.serial.Serial.return_value = printer
    client = booth.app.test_client()
    capture = client.post('/start_time_travel', data={'gender': 'female'})
    assert capture.json == {'job_id': JOB} and finished.wait(5)
    assert client.get(f'/status/{JOB}').json == {
        'status': 'done', 'age': '84.8', 'initial_age': 85, 'target_ages': [75, 90, 90],
        'gpu_job_id': OTHER, 'error': ''}
    photo = client.get(f'/result/{JOB}')
    assert photo.mimetype == 'image/png' and photo.data == png.getvalue()
    seed(booth, OTHER, 'http://booth.test', 0)
    assert client.post(f'/print_result/{JOB}').json == {'job_id': JOB}
    booth.capture_frame.assert_called_once()
    post.assert_called_once()
    booth.serial.Serial.assert_called_once()
    sent = b''.join(call.args[0] for call in write.call_args_list)
    i = sent.index(b'\x1dv0\x00') + 4
    w = int.from_bytes(sent[i:i+2], 'little') * 8
    h = int.from_bytes(sent[i+2:i+4], 'little')
    assert w == 384 and len(sent[i+4:-4]) == (w // 8) * h and sent[-4:] == b'\n'*4
    receipt = Image.frombytes('1', (w, h), sent[i+4:-4], 'raw', '1;I')
    np.testing.assert_array_equal(np.asarray(receipt.crop((0, 45, w, 45+128))), ~dark)
    link = decode(receipt.crop((0, h-w, w, h)))
    assert link == f'http://127.0.0.1:5000/result/{JOB}?download=1'
    assert link == decode(Image.open(io.BytesIO(client.get(f'/qr/{JOB}').data)))
    download = client.get(link)
    assert 'attachment' in download.headers['Content-Disposition'] and download.data == photo.data


def test_live_http_print_and_download(booth, monkeypatch):
    writes = bytearray()

    class Printer(io.BytesIO):
        def write(self, data):
            writes.extend(data)
            return len(data)

    printer = Printer()
    monkeypatch.setattr(booth.serial, "Serial", Mock(return_value=printer))
    server = make_server("127.0.0.1", 0, booth.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        # Fix the public QR payload independently of the OS-assigned HTTP test port.
        expected = seed(booth, JOB, "http://booth.test:5000")
        seed(booth, OTHER, base, 0)  # A newer job must not change the receipt.
        photo = requests.get(f"{base}/result/{JOB}", timeout=3)
        qr = requests.get(f"{base}/qr/{JOB}", timeout=3)
        assert decode(Image.open(io.BytesIO(qr.content))) == expected
        response = requests.post(f"{base}/print_result/{JOB}", timeout=5)
        assert response.status_code == 200 and response.json()["job_id"] == JOB
        assert printer.closed
        start = writes.index(b"\x1dv0\x00") + 4
        w = int.from_bytes(writes[start:start + 2], "little") * 8
        h = int.from_bytes(writes[start + 2:start + 4], "little")
        assert w == 384 and writes[-4:] == b"\n" * 4
        raster = Image.frombytes("1", (w, h), bytes(writes[start + 4:-4]), "raw", "1;I")
        np.testing.assert_array_equal(np.array(raster.crop((0, 45, w, 45 + 128)).convert("L")),
                                      np.array(Image.open(io.BytesIO(photo.content))))
        printed_url = decode(raster.crop((0, h - w, w, h)))
        assert printed_url == expected
        target = urlsplit(printed_url)
        download = requests.get(f"{base}{target.path}?{target.query}", timeout=3)
        assert download.status_code == 200 and download.content == photo.content
        assert "attachment" in download.headers["Content-Disposition"]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("job,state,code", [("invalid", "done", 404), (JOB, None, 404),
                                           (JOB, "processing", 409), (JOB, "error", 409)])
def test_rejects_unprintable_jobs(booth, job, state, code):
    if state:
        booth._set_job(job, status=state)
    client = booth.app.test_client()
    assert client.get(f"/print_result/{job}").status_code == 405
    response = client.post(f"/print_result/{job}")
    assert response.status_code == code and response.json["error"]
    booth.serial.Serial.assert_not_called()


@pytest.mark.parametrize("directory", ["RESULT_DIR", "QR_DIR"])
@pytest.mark.parametrize("corrupt,code", [(False, 404), (True, 500)])
def test_artifact_errors(booth, directory, corrupt, code):
    seed(booth, JOB, "http://booth")
    path = getattr(booth, directory) / f"{JOB}.png"
    if corrupt:
        path.write_bytes(b"not an image")
    else:
        path.unlink()
    response = booth.app.test_client().post(f"/print_result/{JOB}")
    assert response.status_code == code and response.json["error"]
    assert not booth.printer_busy.locked()
    booth.serial.Serial.assert_not_called()


def test_print_contention_and_serial_failure(booth, monkeypatch):
    seed(booth, JOB, "http://booth")
    seed(booth, OTHER, "http://booth")
    writing, release = threading.Event(), threading.Event()

    class BrokenPrinter(io.BytesIO):
        def write(self, data):
            writing.set()
            assert release.wait(5)
            raise OSError("disconnected")

    printer = BrokenPrinter()
    factory = Mock(side_effect=[printer, io.BytesIO()])
    monkeypatch.setattr(booth.serial, "Serial", factory)

    def post(path):
        return booth.app.test_client().post(path)

    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(post, f"/print_result/{JOB}")
        try:
            assert writing.wait(3)
            for path in (f"/print_result/{JOB}", f"/print_result/{OTHER}", "/capture_and_print"):
                assert post(path).status_code == 409
            factory.assert_called_once()
        finally:
            release.set()
        assert active.result().status_code == 500 and printer.closed
    assert post(f"/print_result/{OTHER}").status_code == 200


@pytest.mark.parametrize('failure', ['html', 'jpeg', 'corrupt-png', 'timeout', 'no-face', 'busy',
                                     'fixed-age-api', 'invalid-targets'])
def test_failed_upload_never_announces_receipt(booth, monkeypatch, failure):
    path = booth.CAPTURE_DIR / f'{JOB}.jpg'
    Image.new('RGB', (64, 64)).save(path)
    response = requests.Response()
    response.status_code = 200
    if failure == 'html':
        response._content = b'<html>Relay not ready</html>'
    elif failure == 'corrupt-png':
        data = io.BytesIO()
        Image.new('RGB', (64, 64)).save(data, format='PNG')
        response._content = data.getvalue()[:-20]
    elif failure == 'jpeg':
        response._content = path.read_bytes()
    elif failure in {'fixed-age-api', 'invalid-targets'}:
        data = io.BytesIO()
        Image.new('RGB', (64, 64)).save(data, format='PNG')
        response._content = data.getvalue()
        if failure == 'invalid-targets':
            response.headers.update({'X-Initial-Age': '85', 'X-Target-Ages': '75,95,105'})
    elif failure in {'no-face', 'busy'}:
        response.status_code = 422 if failure == 'no-face' else 409
        response._content = b'{"detail":"retry"}'
    post = Mock(return_value=response, side_effect=requests.Timeout() if failure == 'timeout' else None)
    monkeypatch.setattr(booth.requests, 'post', post)
    booth.run_fading_job(JOB, path, 'female')
    client = booth.app.test_client()
    assert client.get(f'/status/{JOB}').json['status'] == 'error'
    assert client.post(f'/print_result/{JOB}').status_code == 409
    assert not (booth.RESULT_DIR / f'{JOB}.png').exists()
    assert not (booth.QR_DIR / f'{JOB}.png').exists()
