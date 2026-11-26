"""Synthetic photos and a fake serial device only; never operates hardware."""
import ast
import importlib.util
from pathlib import Path
import types
import threading
import io

import pytest
from test_booth_auto_installer import (
    FIXTURE, FakeResponse as LegacyResponse, app_namespace, installer, start,
)

spec = importlib.util.spec_from_file_location(
    'booth_upgrade', Path(__file__).resolve().parents[1] / 'tools/upgrade_booth_ui_qr.py')
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


class FakeResponse(LegacyResponse):
    def __init__(self, status=200):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGB', (1024, 1112), '#aabbcc').save(buffer, 'PNG')
        super().__init__(status=status, content=buffer.getvalue())
        self.headers.update({'X-Initial-Age': '21', 'X-Target-Ages': '11,31,41',
                             'X-Job-Id': 'c' * 32})


def old_app():
    fixture = FIXTURE.replace('<form><button>촬영하고 인쇄하기',
                              '<form method="POST" action="/capture_and_print"><button>촬영하고 인쇄하기')
    fixture = fixture.replace("if __name__ == '__main__':", '''
def process_and_print_image(pil_img, caption=""):
    return "original print"
if __name__ == '__main__':''')
    return installer.build_source(fixture).replace(
        'AUTO_HAI_URL = "http://127.0.0.1:18000"',
        'AUTO_HAI_URL = "http://127.0.0.1:18001"')


@pytest.fixture
def upgraded(app_namespace):
    app, ns = app_namespace
    ns['_booth_original_print_image'] = lambda image, caption='': (image, caption)
    ns['wake_up_and_clear_printer'] = lambda ser: None
    ns['serial'] = types.SimpleNamespace(Serial=lambda *a, **k: pytest.fail('No hardware'))
    ns['_auto_threading'].Lock = threading.Lock
    exec(upgrade.PRINT_EXTENSION, ns)
    tree = ast.parse(upgrade.build_source(old_app()))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_auto_worker', 'auto_booth_status'):
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'synthetic.py', 'exec'), ns)
    app.view_functions['auto_booth_status'] = ns['auto_booth_status']
    return app, ns


def test_upgrade_preserves_address_and_original_functions():
    source = old_app()
    updated = upgrade.build_source(source)
    assert 'AUTO_HAI_URL = "http://127.0.0.1:18001"' in updated
    old_tree, new_tree = ast.parse(source), ast.parse(updated)
    old_capture = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'capture_frame')
    new_capture = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'capture_frame')
    assert ast.dump(old_capture) == ast.dump(new_capture)
    assert 'def _booth_original_print_image' in updated
    assert '/capture_only' in updated and '/capture_and_print' in updated
    assert '사진을 Runpod으로' in updated
    assert '10년 전·현재 원본·10년 후·20년 후' in updated
    with pytest.raises(ValueError, match='이미 적용'):
        upgrade.build_source(updated)
    with pytest.raises(ValueError, match='기존 자동'):
        upgrade.build_source(FIXTURE)


def test_print_once_and_refresh_never_prints(upgraded, monkeypatch):
    app, ns = upgraded
    monkeypatch.setattr(ns['_auto_requests'], 'post', lambda *a, **k: FakeResponse())
    receipts = []
    ns['_booth_send_receipt'] = lambda image: receipts.append(image)
    client = app.test_client()
    job_id = start(client).json['job_id']
    for _ in range(3):
        state = client.get('/api/booth/jobs/' + job_id).json
        assert state['status'] == 'done'
        assert state['qr_print_status'] == 'sent'
        assert state['target_ages'] == [11, 31, 41]
        assert state['initial_age'] == 21
        assert state['gpu_job_id'] == 'c' * 32
        assert state['photo_captured'] is True
        assert client.get(state['qr_url']).status_code == 200
    assert len(receipts) == 1
    assert receipts[0].height > 700  # Four-cut photograph and the QR are on one receipt.
    state = ns['_auto_read'](job_id)
    ns['_booth_print_qr_once'](job_id, state)
    assert len(receipts) == 1
    assert not ns['capture_busy'].locked()


def test_print_failure_keeps_download(upgraded, monkeypatch):
    app, ns = upgraded
    monkeypatch.setattr(ns['_auto_requests'], 'post', lambda *a, **k: FakeResponse())
    def fail(image):
        raise OSError('synthetic unplugged printer')
    ns['_booth_send_receipt'] = fail
    client = app.test_client()
    job_id = start(client).json['job_id']
    state = client.get('/api/booth/jobs/' + job_id).json
    assert state['status'] == 'done'
    assert state['qr_print_status'] == 'error'
    assert client.get(state['download_url']).status_code == 200
    assert not ns['capture_busy'].locked()


def test_restart_during_print_does_not_stall_or_reprint(upgraded, monkeypatch):
    app, ns = upgraded
    monkeypatch.setattr(ns['_auto_requests'], 'post', lambda *a, **k: FakeResponse())
    ns['_booth_send_receipt'] = lambda image: None
    client = app.test_client()
    job_id = start(client).json['job_id']
    state = ns['_auto_read'](job_id)
    state.update(qr_print_status='printing', run_id='previous-process')
    ns['_auto_write'](job_id, state)
    status = client.get('/api/booth/jobs/' + job_id).json
    assert status['status'] == 'done'
    assert status['qr_print_status'] == 'unknown'


def test_qr_receipt_decodes_without_photo_filter(upgraded):
    import cv2
    import numpy as np
    _, ns = upgraded
    url = 'http://192.168.1.50:5000/auto-results/' + 'a' * 32 + '.png?download=1'
    image = ns['_booth_qr_receipt'](url)
    assert image.width == 384
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(np.array(image.convert('L')))
    assert decoded == url


def test_serial_payload_and_original_print_wrapper(upgraded):
    _, ns = upgraded
    writes, options = [], {}
    class FakeSerial:
        def __init__(self, *args, **kwargs):
            options.update(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def write(self, chunk):
            writes.append(chunk)
            return len(chunk)
        def flush(self):
            pass
    ns['serial'] = types.SimpleNamespace(Serial=FakeSerial)
    ns['_booth_send_receipt'](ns['_booth_qr_receipt']('https://example.org/synthetic'))
    assert writes[0][:6] == bytes((0x1d, 0x76, 0x30, 0, 48, 0))
    assert writes[-1] == b'\n\n\n\n'
    assert options['write_timeout'] == 15
    assert ns['process_and_print_image']('synthetic', 'caption') == ('synthetic', 'caption')


def test_install_backs_up_without_opening_printer(tmp_path, monkeypatch):
    import serial
    monkeypatch.setattr(serial, 'Serial', lambda *a, **k: pytest.fail('No print during install'))
    path = tmp_path / 'app.py'
    source = old_app()
    path.write_text(source)
    path.chmod(0o640)
    backup = upgrade.install(path)
    assert backup.read_text() == source
    assert path.stat().st_mode & 0o777 == 0o640
    assert upgrade.MARKER in path.read_text()


def test_partial_serial_write_closes_device_and_never_retries(upgraded):
    _, ns = upgraded
    calls = []
    class ShortSerial:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            calls.append('closed')
        def write(self, chunk):
            calls.append('write')
            return len(chunk) - 1
    ns['serial'] = types.SimpleNamespace(Serial=ShortSerial)
    with pytest.raises(RuntimeError, match='중간에'):
        ns['_booth_send_receipt'](ns['_booth_qr_receipt']('https://example.org/synthetic'))
    assert calls == ['write', 'closed']
    assert not ns['_booth_printer_lock'].locked()


@pytest.mark.parametrize('headers', [{}, {'X-Initial-Age': '21', 'X-Target-Ages': '10,30,50', 'X-Job-Id': 'c'*32}])
def test_rejects_legacy_or_wrong_timeline_without_printing(upgraded, monkeypatch, headers):
    app, ns = upgraded
    response = FakeResponse()
    response.headers = headers
    monkeypatch.setattr(ns['_auto_requests'], 'post', lambda *a, **k: response)
    ns['_booth_send_receipt'] = lambda image: pytest.fail('Must not print an unverified timeline')
    client = app.test_client()
    job_id = start(client).json['job_id']
    state = client.get('/api/booth/jobs/' + job_id).json
    assert state['status'] == 'error'
    assert client.get('/auto-results/' + job_id + '.png').status_code == 409
    assert not ns['capture_busy'].locked()


def test_combined_receipt_qr_and_photo_match(upgraded):
    import cv2
    import numpy as np
    from PIL import Image
    _, ns = upgraded
    url = 'http://192.168.1.50:5000/auto-results/' + 'a'*32 + '.png?download=1'
    receipt = ns['_booth_qr_receipt'](url, Image.new('RGB', (1024, 1112), 'black'), 21, [11,31,41])
    assert receipt.getpixel((100, 100)) == 0  # The supplied four-cut, not a new camera frame.
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(np.array(receipt.convert('L')))
    assert decoded == url
