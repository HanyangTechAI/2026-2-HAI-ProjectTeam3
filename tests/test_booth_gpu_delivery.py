"""Opt-in engineering check, not a hardware or model-quality trial.

Set BOOTH_GPU_TEST_PORT, BOOTH_GPU_TEST_IMAGE and BOOTH_GPU_TEST_GENDER to
exercise the real loopback GPU service with an internally permitted photo.
Only webcam and printer are simulated. The GPU service retains private
input/output under its normal retention policy; do not use visitor photos.
"""
import importlib.util
import io
import os
import tempfile
import threading
from pathlib import Path
from time import monotonic
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests
from PIL import Image, ImageEnhance, ImageFilter
from werkzeug.serving import make_server


@pytest.mark.skipif(not os.getenv('BOOTH_GPU_TEST_PORT'), reason='requires live GPU API')
def test_live_gpu_delivery(monkeypatch):
    api = f"http://127.0.0.1:{int(os.environ['BOOTH_GPU_TEST_PORT'])}"
    health = requests.get(f'{api}/health', timeout=5).json()
    assert health['real_age_model'] and health['target_offsets'] == [-10, 10, 20]
    frame = cv2.imread(os.environ['BOOTH_GPU_TEST_IMAGE'])
    gender = os.environ['BOOTH_GPU_TEST_GENDER']
    assert frame is not None and gender in {'female', 'male'}
    with tempfile.TemporaryDirectory() as home:
        monkeypatch.setenv('HOME', home)
        monkeypatch.setenv('PUBLIC_HOST', '127.0.0.1')
        monkeypatch.setattr(cv2, 'VideoCapture', Mock())
        path = Path(__file__).resolve().parents[1] / 'raspi/photobooth_app.py'
        spec = importlib.util.spec_from_file_location('live_booth', path)
        booth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(booth)
        booth.capture_frame = Mock(return_value=frame)
        booth.FADING_API_URL = f'{api}/api/fading/fourcut'
        sent = bytearray()

        class SimulatedPrinter(io.BytesIO):
            def write(self, data):
                sent.extend(data)
                return len(data)

        monkeypatch.setattr(booth.serial, 'Serial', Mock(return_value=SimulatedPrinter()))
        done = threading.Event()
        worker = booth.run_fading_job

        def observed_worker(*args):
            try:
                worker(*args)
            finally:
                done.set()

        booth.run_fading_job = observed_worker
        server = make_server('127.0.0.1', 0, booth.app, threaded=True)
        booth.WEB_PORT = server.server_port
        base = f'http://127.0.0.1:{server.server_port}'
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            started = monotonic()
            capture = requests.post(f'{base}/start_time_travel', data={'gender': gender}, timeout=10)
            capture.raise_for_status()
            job = capture.json()['job_id']
            assert done.wait(booth.API_TIMEOUT + 15), 'Pi worker did not finish'
            status = requests.get(f'{base}/status/{job}', timeout=5).json()
            assert status['status'] == 'done', status.get('error')
            age = status['initial_age']
            assert status['target_ages'] == [max(3, min(90, age + d)) for d in (-10, 10, 20)]
            assert booth.ID_RE.fullmatch(status['gpu_job_id'])
            display = requests.get(f'{base}/result/{job}', timeout=5)
            display.raise_for_status()
            with Image.open(io.BytesIO(display.content)) as sheet:
                assert (sheet.format, sheet.mode, sheet.size) == ('PNG', 'RGB', (1024, 1112))
                scaled = sheet.resize((384, int(384 * sheet.height / sheet.width)), Image.Resampling.LANCZOS)
                mono = scaled.convert('L').filter(ImageFilter.SHARPEN)
                mono = ImageEnhance.Brightness(mono).enhance(1.25)
                expected = ImageEnhance.Contrast(mono).enhance(1.3).convert('1')
            ready = monotonic() - started
            printed = requests.post(f'{base}/print_result/{job}', timeout=15)
            printed.raise_for_status()
            assert printed.json()['job_id'] == job
            booth.capture_frame.assert_called_once()
            i = sent.index(b'\x1dv0\x00') + 4
            w = int.from_bytes(sent[i:i+2], 'little') * 8
            h = int.from_bytes(sent[i+2:i+4], 'little')
            assert w == 384
            paper = Image.frombytes('1', (w, h), bytes(sent[i+4:-4]), 'raw', '1;I')
            assert paper.crop((0, 45, w, 45 + expected.height)).tobytes() == expected.tobytes()
            qr_area = np.array(paper.crop((0, h-w, w, h)).convert('L'))
            link = cv2.QRCodeDetector().detectAndDecode(qr_area)[0]
            assert link == f'{base}/result/{job}?download=1'
            screen_qr = requests.get(f'{base}/qr/{job}', timeout=5)
            screen_qr.raise_for_status()
            with Image.open(io.BytesIO(screen_qr.content)) as qr:
                assert cv2.QRCodeDetector().detectAndDecode(np.array(qr.convert('L')))[0] == link
            download = requests.get(link, timeout=5)
            download.raise_for_status()
            assert download.content == display.content
            print(f"Live GPU {status['gpu_job_id']} + simulated Pi {job}: "
                  f"{age} -> {status['target_ages']}; PNG {ready:.1f}s; QR download {monotonic()-started:.1f}s")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
