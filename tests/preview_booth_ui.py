"""Optional hardware-free preview. All photos, ages and printer replies are fake."""
import ast
import io
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_booth_ui_qr_upgrade import old_app, upgrade
from flask import Flask, Response, jsonify, render_template_string, request, send_file, abort
from PIL import Image, ImageDraw

app = Flask(__name__)
source = ast.parse(upgrade.build_source(old_app()))
template = next(ast.literal_eval(n.value) for n in source.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HTML_TEMPLATE' for t in n.targets))
template = template.replace('<main class="card" id="main">', '<main class="card" id="main"><p class="note" style="background:#fff0ce;padding:10px">합성 자료로 보는 화면 시험이에요. 실제 촬영·생성·인쇄는 하지 않아요.</p>')
template = template.replace('<h1>AI 타임머신 포토부스</h1>', '''
<section aria-labelledby="install-heading" style="border:2px solid #005bc4;padding:16px;border-radius:12px;margin:16px 0 24px">
<h2 id="install-heading" style="margin:0">설치 파일 받기</h2>
<p class="note">먼저 Runpod용 파일을 내려받아 주세요. 다운로드만으로 설치되지는 않아요.</p>
<a class="download" href="/installer-download/upgrade_booth_relative_runpod.py" download>Runpod 설치 파일 다운로드</a>
<details><summary>라즈베리파이 파일·전체 안내</summary>
<a class="download" href="/installer-download/upgrade_booth_relative_pi.py" download>라즈베리파이 설치 파일 다운로드</a>
<a class="download" href="/installer-download/booth_relative_setup.md" download>적용 순서 안내 다운로드</a>
</details></section><h1>AI 타임머신 포토부스</h1>
''')
jobs = {}


@app.get('/installer-download/<filename>')
def installer_download(filename):
    if filename not in {'upgrade_booth_relative_runpod.py', 'upgrade_booth_relative_pi.py', 'booth_relative_setup.md'}:
        abort(404)
    path = Path(__file__).resolve().parents[1] / '.baze-files' / 'outputs' / filename
    if not path.is_file():
        abort(404)
    response = send_file(path, as_attachment=True, download_name=filename, mimetype='application/octet-stream')
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


def png(width, height, title):
    image = Image.new('RGB', (width, height), '#dce7f3')
    draw = ImageDraw.Draw(image)
    draw.text((25, 25), title, fill='#203b5b')
    draw.ellipse((width*.32, height*.18, width*.68, height*.63), fill='#9fb8d3')
    draw.rounded_rectangle((width*.15, height*.65, width*.85, height*1.1), radius=40, fill='#7c9bbd')
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    return buffer.getvalue()


@app.get('/')
def home():
    return render_template_string(template)


@app.get('/video_feed')
def video():
    return Response(png(640, 480, 'SYNTHETIC CAMERA PREVIEW'), mimetype='image/png')


@app.post('/api/booth/jobs')
def start():
    data = request.get_json()
    if data.get('consent') is not True or data.get('gender') not in ('female', 'male'):
        return jsonify(error='조건과 동의가 필요해요.'), 400
    job_id = uuid.uuid4().hex
    jobs[job_id] = time.time()
    return jsonify(job_id=job_id), 202


@app.get('/api/booth/jobs/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify(error='촬영 기록을 찾지 못했어요.'), 404
    created = jobs[job_id]
    done = time.time() - created > 6
    return jsonify(status='done' if done else 'processing', created=created, photo_captured=True,
                   message='합성 자료로 처리 화면을 시험하고 있어요.', estimated_age=24.2,
                   total_seconds=6.0, qr_print_status='sent',
                   initial_age=24, target_ages=[14,34,44],
                   qr_print_message='모의 인쇄 전송 결과예요. 실제 출력은 하지 않았어요.',
                   image_url='/synthetic.png', download_url='/synthetic.png', qr_url='/synthetic-qr.png')


@app.get('/synthetic.png')
def result():
    return Response(png(700, 700, 'SYNTHETIC FOUR-CUT PLACEHOLDER'), mimetype='image/png')


@app.get('/synthetic-qr.png')
def qr():
    import qrcode
    buffer = io.BytesIO()
    qrcode.make('https://example.org/synthetic-preview-only').save(buffer, format='PNG')
    return Response(buffer.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5173, debug=True, use_debugger=False)
