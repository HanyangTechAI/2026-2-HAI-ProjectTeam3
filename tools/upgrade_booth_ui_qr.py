#!/usr/bin/env python3
"""Upgrade the installed v1 booth without replacing its working backend.

UI and relative result handling adapted from team PRs #2/#3.
Run with the Pi booth's own Python, after updating the Runpod program.
Never starts services or deletes photos. No print is sent during installation.
"""
import argparse
import ast
from datetime import datetime
from pathlib import Path
import re
import shutil

MARKER = '# booth-relative-receipt-v3'

HTML = r'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 타임머신 포토부스</title>
<meta name="description" content="촬영한 사진으로 목표 연령별 4컷을 만들고 QR로 받아요.">
<meta name="theme-color" content="#f2f2f7">
<style>
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:#f2f2f7;color:#1c1c1e;font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif;word-break:keep-all;line-height:1.75}
.card{max-width:520px;margin:auto;background:white;border-radius:18px;padding:24px;box-shadow:0 10px 25px #00000012}h1,h2{letter-spacing:-.02em;line-height:1.35}h1{font-size:25px;margin:0 0 8px;color:#005bc4}h2{font-size:21px}p{margin:10px 0}.note{font-size:14px;color:#505562}.stage{position:relative;aspect-ratio:4/3;background:#131722;border-radius:14px;overflow:hidden;margin:18px 0}.stage img{width:100%;height:100%;object-fit:cover;display:block}
#countdown{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#0006;color:white;font-size:100px;font-weight:800;text-align:center;line-height:1.4}#countdown.small{font-size:26px;padding:20px}label{display:block;margin:16px 0 6px;font-size:15px;font-weight:600}select,textarea{width:100%;padding:12px;border:1px solid #858997;border-radius:10px;background:white;font:inherit;color:inherit}textarea{height:80px}button,.download{display:block;min-height:48px;padding:13px 16px;border:0;border-radius:12px;font:inherit;font-weight:700;text-align:center;cursor:pointer;background:#005bc4;color:white;width:100%;margin-top:12px}button:disabled{background:#626d7b;cursor:wait}.consent{display:flex;align-items:flex-start;gap:10px;font-weight:400;font-size:14px;text-align:left}.consent input{width:22px;height:22px;flex-shrink:0;margin-top:3px}.elapsed{font-size:38px;font-weight:800;color:#005bc4;font-variant-numeric:tabular-nums}#processing{text-align:center;padding:35px 0}.spin{width:44px;height:44px;border:4px solid #d1d1d6;border-top-color:#005bc4;border-radius:50%;margin:20px auto;animation:spin 1.2s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}#result-img{display:block;width:100%;height:auto}#qr-img{display:block;width:240px;max-width:100%;height:auto;margin:16px auto}.error{color:#a51e24;background:#fff0f1;padding:12px;border-radius:8px}details{margin-top:25px;border-top:1px solid #d1d1d6;padding-top:15px}summary{cursor:pointer;min-height:44px;font-weight:600}a{color:#0053ad}button,input,select,summary,a{touch-action:manipulation}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #12324e;outline-offset:3px}[hidden]{display:none!important}@media(prefers-reduced-motion:reduce){.spin{animation:none}}@media(max-width:400px){.card{padding:18px}h1{font-size:22px}}
button:hover:not(:disabled),.download:hover{background:#004695}summary:hover{text-decoration:underline}h1,h2{text-wrap:balance}.skip{position:absolute;left:16px;top:-100px;background:white;padding:8px;z-index:2}.skip:focus{top:8px}html{-webkit-tap-highlight-color:#005bc433}
</style></head><body><a href="#main" class="skip">촬영 화면으로 건너뛰기</a><main class="card" id="main">
<h1>AI 타임머신 포토부스</h1>
<section id="idle">
<p class="note">사진 한 장으로 만나는 다양한 나이의 모습이에요.</p>
<div class="stage"><img id="preview" src="/video_feed" width="640" height="480" alt="좌우반전된 카메라 미리보기"><div id="countdown" role="status" hidden></div></div>
<form id="travel-form" autocomplete="off">
<label for="gender">연령변환 성별 조건 *</label><select id="gender" name="gender" required><option value="">선택하세요</option><option value="female">여성</option><option value="male">남성</option></select>
<label class="consent" for="consent"><input id="consent" name="consent" type="checkbox" required><span>사진을 Runpod으로 보내 연령변환하고 사진·결과를 Runpod과 라즈베리파이에 저장하는 데 동의해요. 파일은 자동 삭제되지 않아요.</span></label>
<p class="note">추정 나이 기준 10년 전·현재 원본·10년 후·20년 후의 4컷을 만들어요. 편집 목표는 3~90세로 제한하며 실제 과거·미래 얼굴 예측은 아니에요.</p>
<button id="start" type="submit">시간 여행 시작</button>
</form>
<p class="note">버튼을 누르면 3초 뒤 촬영해요. 완성되면 4컷 사진과 다운로드 QR을 한 장에 자동으로 한 번 인쇄해요.</p>
<details id="legacy"><summary>기존 원본 촬영·저장·인쇄</summary>__LEGACY_FORMS__</details>
__LEGACY_RESULTS__
</section>
<section id="processing" hidden><div class="spin" aria-hidden="true"></div><h2>시간 여행 중이에요</h2><p id="progress" role="status" aria-live="polite">처리 상태를 확인하고 있어요.</p><p class="elapsed" id="elapsed">00:00</p><p class="note">생성에는 수 분이 걸릴 수 있어요.<br>다시 촬영하지 말고 기다려 주세요.</p></section>
<section id="done" hidden><h2>시간 여행 완료!</h2><p class="note" id="metadata"></p><p class="note" id="timeline"></p><img id="result-img" width="1024" height="1112" alt="현재 원본과 목표 연령별 변환 결과 4컷"><a class="download" id="download">4컷 다운로드</a><h2>휴대전화로 받기</h2><img id="qr-img" width="240" height="240" alt="완성된 4컷 다운로드 QR 코드"><p class="note">같은 와이파이에서 스캔해 주세요. 완성 후 1시간 동안 다운로드할 수 있어요.</p><p id="print-note" role="status"></p><button type="button" id="again">다음 사람 촬영</button></section>
<p id="error" class="error" role="alert" hidden></p><button id="retry" type="button" hidden>처리 상태 다시 확인</button>
</main>
__LEGACY_SCRIPT__
<script>
(()=>{
const $=id=>document.getElementById(id), key='photobooth-auto-current-job-v1';
let running=false, active=null, timer=null, started=0;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function saved(){try{return sessionStorage.getItem(key)}catch(_){return null}}
function save(id){try{id?sessionStorage.setItem(key,id):sessionStorage.removeItem(key)}catch(_){}}
function show(id){['idle','processing','done'].forEach(s=>$(s).hidden=s!==id)}
function busy(value){running=value;document.querySelectorAll('#idle button,#idle input,#idle select,#idle textarea').forEach(e=>e.disabled=value)}
function clock(){const s=Math.max(0,Math.floor(Date.now()/1000-started));$('elapsed').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
async function json(url,options={}){const c=new AbortController(), t=setTimeout(()=>c.abort(),15000);try{const r=await fetch(url,{...options,signal:c.signal,cache:'no-store'});let d;try{d=await r.json()}catch(_){throw Error('응답을 읽지 못했어요. 연결을 확인해 주세요.')}if(!r.ok){const e=Error(d.error||'요청을 처리하지 못했어요.');e.status=r.status;throw e}return d}finally{clearTimeout(t)}}
function failed(e){clearInterval(timer);timer=null;if(e.status===404||e.status===410){save(null);active=null;$('result-img').removeAttribute('src');$('qr-img').removeAttribute('src');show('idle')}$('countdown').hidden=true;$('error').textContent=e.name==='AbortError'?'연결 응답이 늦어요. 새 촬영 대신 처리 상태를 다시 확인해 주세요.':e.message;$('error').hidden=false;const pending=Boolean(saved()||active);$('retry').hidden=!pending;busy(pending)}
async function poll(id){show('processing');let failures=0;const began=Date.now();while(Date.now()-began<12*60*1000){let j;try{j=await json('/api/booth/jobs/'+encodeURIComponent(id));failures=0;$('error').hidden=true}catch(e){if(e.status===404||e.status===410)throw e;failures++;$('error').textContent='연결이 끊겼어요. 자동으로 다시 확인하고 있어요. 새로 촬영하지 마세요.';$('error').hidden=false;if(failures>=4)throw e;await sleep(2000);continue}started=j.created;if(!timer)timer=setInterval(clock,1000);clock();$('progress').textContent=j.message;if(j.status==='processing'&&!j.photo_captured){show('idle');$('countdown').classList.add('small');$('countdown').textContent='촬영 중이에요. 움직이지 마세요.';$('countdown').hidden=false}else{$('countdown').hidden=true;show('processing')}if(j.status==='error'){save(null);active=null;show('idle');throw Error(j.message)}if(j.status==='done'){
if(j.qr_print_status==='printing'){ $('progress').textContent='4컷과 QR이 완성돼 영수증을 인쇄하고 있어요.';await sleep(1000);continue }
clearInterval(timer);timer=null;$('result-img').src=j.image_url;$('qr-img').src=j.qr_url;$('download').href=j.download_url;$('metadata').textContent='추정 나이: '+(j.estimated_age==null?'확인되지 않음':j.estimated_age.toFixed(1)+'세')+' · 촬영 요청부터 QR 완성까지 '+j.total_seconds.toFixed(1)+'초';$('timeline').textContent=Array.isArray(j.target_ages)&&j.target_ages.length===3?j.target_ages[0]+'세 편집 → 현재 원본(약 '+j.initial_age+'세) → '+j.target_ages[1]+'세 편집 → '+j.target_ages[2]+'세 편집':'이전 고정 연령 구성의 결과예요. 새 촬영부터 상대 연령으로 만들어요.';$('print-note').textContent=j.qr_print_message||'이전 촬영 결과예요. 자동 인쇄는 새 촬영부터 적용돼요.';show('done');busy(false);return}await sleep(2000)}throw Error('처리가 오래 걸려요. 새 촬영 대신 처리 상태를 다시 확인해 주세요.')}
async function resume(){const id=saved()||active;if(!id)return;busy(true);$('retry').hidden=true;$('error').hidden=true;try{await poll(id)}catch(e){failed(e)}}
$('travel-form').addEventListener('submit',async e=>{e.preventDefault();if(running||!$('travel-form').reportValidity())return;const gender=$('gender').value;save(null);active=null;busy(true);$('error').hidden=true;$('retry').hidden=true;try{const cd=$('countdown');cd.classList.remove('small');cd.hidden=false;for(let n=3;n>=1;n--){cd.textContent=n;await sleep(1000)}cd.classList.add('small');cd.textContent='촬영 중이에요.\n움직이지 마세요.';const j=await json('/api/booth/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gender,consent:true})});active=j.job_id;save(active);cd.hidden=true;await poll(active)}catch(err){failed(err)}});
$('retry').addEventListener('click',()=>{clearInterval(timer);timer=null;resume()});$('again').addEventListener('click',()=>{save(null);active=null;location.reload()});resume();
})();
</script></body></html>'''

PRINT_EXTENSION = r'''
# booth-relative-receipt-v3
_booth_printer_lock = _auto_threading.Lock()


def process_and_print_image(pil_img, caption=""):
    with _booth_printer_lock:
        return _booth_original_print_image(pil_img, caption)


def _booth_qr_receipt(url, photo=None, initial_age=None, targets=None):
    # Integer module size, four-module quiet zone, no filtering or dithering.
    qr = _auto_qrcode.QRCode(
        error_correction=_auto_qrcode.constants.ERROR_CORRECT_M,
        box_size=1, border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    modules = qr.modules_count + 8
    scale = min(6, 352 // modules)
    if scale < 3:
        raise RuntimeError("QR 주소가 너무 길어 인쇄하지 않았어요.")
    qr.box_size = scale
    code = qr.make_image(fill_color="black", back_color="white").convert("1")
    photo_height = 0
    if photo is not None:
        from PIL import ImageEnhance as _QrEnhance, ImageFilter as _QrFilter
        if photo.size != (1024, 1112):
            raise RuntimeError("4컷 사진 크기가 예상과 달라 인쇄하지 않았어요.")
        photo_height = round(384 * photo.height / photo.width)
        photo = photo.resize((384, photo_height), Image.Resampling.LANCZOS).convert("L")
        photo = photo.filter(_QrFilter.SHARPEN)
        photo = _QrEnhance.Brightness(photo).enhance(1.25)
        photo = _QrEnhance.Contrast(photo).enhance(1.3)
        photo = photo.convert("1", dither=Image.Dither.FLOYDSTEINBERG)
    qr_top = 80 + photo_height
    receipt = Image.new("1", (384, qr_top + code.height + 72), 1)
    if photo is not None:
        receipt.paste(photo, (0, 64))
    receipt.paste(code, ((384 - code.width) // 2, qr_top))
    # PIL's built-in Latin font is available even without Korean font packages.
    from PIL import ImageDraw as _QrDraw
    draw = _QrDraw.Draw(receipt)
    draw.text((20, 12), "AI TIME MACHINE - PHOTO + QR", fill=0)
    if initial_age is not None and targets:
        draw.text((20, 32), f"Ages: {targets[0]} / {initial_age} (photo) / {targets[1]} / {targets[2]}", fill=0)
    draw.text((20, qr_top + code.height + 12), "Same Wi-Fi / Valid for 1 hour", fill=0)
    return receipt


def _booth_send_receipt(image):
    packed = bytearray()
    for y in range(image.height):
        for x in range(0, 384, 8):
            value = 0
            for bit in range(8):
                if image.getpixel((x + bit, y)) == 0:
                    value |= 1 << (7 - bit)
            packed.append(value)
    header = bytes((0x1d, 0x76, 0x30, 0, 48, 0,
                    image.height & 255, image.height >> 8))
    with _booth_printer_lock:
        with serial.Serial("/dev/ttyACM0", 115200, timeout=2, write_timeout=15) as ser:
            wake_up_and_clear_printer(ser)
            payload = header + packed
            for offset in range(0, len(payload), 1024):
                chunk = payload[offset:offset + 1024]
                if ser.write(chunk) != len(chunk):
                    raise RuntimeError("프린터 전송이 중간에 멈췄어요.")
                time.sleep(0.01)
            if ser.write(b"\n\n\n\n") != 4:
                raise RuntimeError("프린터 전송이 중간에 멈췄어요.")
            ser.flush()


def _booth_print_qr_once(job_id, state):
    # Only the generation worker calls this; browsing/reloads never print.
    if state.get("qr_print_status") != "printing":
        return
    try:
        with Image.open(AUTO_ROOT / job_id / "fourcut.png") as image:
            receipt = _booth_qr_receipt(state["public_download_url"], image.convert("RGB"),
                                        state["initial_age"], state["target_ages"])
        _booth_send_receipt(receipt)
        state.update(qr_print_status="sent", qr_print_message=
            "4컷과 QR을 프린터로 전송했어요. 종이 출력과 스캔을 확인해 주세요.")
    except Exception:
        app.logger.exception("QR printing failed: %s", job_id)
        state.update(qr_print_status="error", qr_print_message=
            "4컷·QR 인쇄를 완료했는지 확인할 수 없어요. 용지·연결을 확인하고 화면 QR을 이용해 주세요. 자동 재인쇄는 하지 않아요.")
    _auto_write(job_id, state)
'''


def build_source(source):
    if MARKER in source:
        raise ValueError('이미 적용된 버전이에요. 중복 설치하지 않았어요.')
    if '# booth-ui-qr-print-v2' in source:
        raise ValueError('앞서 준비한 QR 전용 버전이 이미 설치돼 있어요. 덮어쓰지 않았으니 알려주세요.')
    if '# photobooth-auto-qr-v1' not in source:
        raise ValueError('기존 자동 4컷·QR 버전이 필요해요.')
    tree = ast.parse(source)
    templates = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'HTML_TEMPLATE' for t in n.targets)]
    printers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_and_print_image']
    if len(templates) != 1 or len(printers) != 1:
        raise ValueError('기존 화면·인쇄 구성이 예상과 달라 중단했어요.')
    old = ast.literal_eval(templates[0].value)
    forms = []
    for action in ('capture_only', 'capture_and_print'):
        found = re.findall(r'<form\b[^>]*action="/' + action + r'"[^>]*>.*?</form>', old, re.S)
        if len(found) != 1:
            raise ValueError('기존 원본 촬영·인쇄 양식이 예상과 달라 중단했어요.')
        forms.extend(found)
    original_scripts = [s for s in re.findall(r'<script>.*?</script>', old, re.S)
                        if "photobooth-auto-current-job-v1" not in s]
    results = re.findall(r'{% if captured_name %}.*?{% endif %}', old, re.S)
    html = HTML.replace('__LEGACY_FORMS__', '\n'.join(forms))
    html = html.replace('__LEGACY_SCRIPT__', '\n'.join(original_scripts))
    html = html.replace('__LEGACY_RESULTS__', '\n'.join(results) +
                        '{% if status %}<p role="status">{{ status }}</p>{% endif %}')
    # Give the original optional caption a real label, preserving its form.
    html = html.replace('<textarea name="caption"',
                        '<label for="caption">원본 인쇄 메시지 (선택)</label><textarea id="caption" name="caption"')
    lines = source.splitlines(keepends=True)
    node = templates[0]
    lines[node.lineno - 1:node.end_lineno] = ['HTML_TEMPLATE = ' + repr(html) + '\n']
    updated = ''.join(lines)
    updated, count = re.subn(r'^def process_and_print_image\(', 'def _booth_original_print_image(', updated, flags=re.M)
    if count != 1:
        raise ValueError('기존 인쇄 함수를 보존하지 못했어요.')
    anchor = '        _auto_write(job_id, state)\n    except _auto_requests.Timeout:'
    if updated.count(anchor) != 1:
        raise ValueError('완료 처리 위치가 달라 중단했어요.')
    updated = updated.replace(anchor,
        '        state.update(qr_print_status="printing", qr_print_message="4컷과 QR을 인쇄하고 있어요.")\n'
        '        _auto_write(job_id, state)\n'
        '        _booth_print_qr_once(job_id, state)\n    except _auto_requests.Timeout:', 1)
    anchor = '            estimated_age=state.get("estimated_age"),'
    if updated.count(anchor) != 1:
        raise ValueError('상태 응답 구성이 달라 중단했어요.')
    updated = updated.replace(anchor, anchor + '\n'
        '            qr_print_status=state.get("qr_print_status", "legacy"),\n'
        '            qr_print_message=state.get("qr_print_message", ""),\n'
        '            initial_age=state.get("initial_age"),\n'
        '            target_ages=state.get("target_ages"),\n'
        '            gpu_job_id=state.get("gpu_job_id"),', 1)
    anchor = '            estimated = _auto_number(response.headers.get("X-Estimated-Age"))'
    if updated.count(anchor) != 1:
        raise ValueError('생성 결과 확인 구성이 달라 중단했어요.')
    updated = updated.replace(anchor, anchor + '''
            try:
                initial_age = int(response.headers["X-Initial-Age"])
                targets = [int(value) for value in response.headers["X-Target-Ages"].split(",")]
                expected = [max(3, min(90, initial_age + d)) for d in (-10, 10, 20)]
                gpu_job_id = response.headers["X-Job-Id"]
                if not 1 <= initial_age <= 100 or targets != expected or not _auto_id_valid(gpu_job_id):
                    raise ValueError("Unexpected timeline")
            except (KeyError, ValueError):
                raise RuntimeError("상대 연령 결과를 확인하지 못했어요. Runpod 변경·재실행 여부를 확인해 주세요. 기존 고정 연령 결과를 상대 연령으로 표시하거나 인쇄하지 않았어요.")
            state.update(initial_age=initial_age, target_ages=targets, gpu_job_id=gpu_job_id)
''', 1)
    anchor = '            if image.format != "PNG" or image.width * image.height > 20_000_000:'
    if updated.count(anchor) != 1:
        raise ValueError('사진 검증 구성이 달라 중단했어요.')
    updated = updated.replace(anchor,
        '            if image.format != "PNG" or image.size != (1024, 1112):', 1)
    anchor = '    result = {key: state[key] for key in ("status", "message", "created")}'
    if updated.count(anchor) != 1:
        raise ValueError('상태 확인 위치가 달라 중단했어요.')
    updated = updated.replace(anchor,
        '    if state.get("qr_print_status") == "printing" and state.get("run_id") != AUTO_RUN_ID:\n'
        '        state.update(qr_print_status="unknown", qr_print_message="인쇄 중 프로그램이 재시작됐어요. 출력 여부를 확인하고 화면 QR을 이용해 주세요. 자동 재인쇄는 하지 않아요.")\n' + anchor, 1)
    updated = updated.replace(anchor, anchor + '\n    result["photo_captured"] = bool(state.get("photo_captured"))', 1)
    anchor = '        state["message"] = "HAI에서 나이 추정과 연령변환을 진행하고 있어요"'
    if updated.count(anchor) != 1:
        raise ValueError('촬영 완료 확인 위치가 달라 중단했어요.')
    updated = updated.replace(anchor, '        state["photo_captured"] = True\n' + anchor, 1)
    tree = ast.parse(updated)
    mains = [n for n in tree.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == "__name__ == '__main__'"]
    if len(mains) != 1:
        raise ValueError('실행 구성을 확인하지 못했어요.')
    lines = updated.splitlines(keepends=True)
    lines[mains[0].lineno - 1:mains[0].lineno - 1] = ['\n' + PRINT_EXTENSION + '\n']
    updated = ''.join(lines)
    # Preserve AUTO_HAI_URL exactly; UI terminology alone now says Runpod.
    updated = updated.replace('HAI에서', 'Runpod에서').replace('HAI가', 'Runpod이').replace('HAI에', 'Runpod에').replace('HAI ', 'Runpod ')
    compile(updated, 'photobooth/app.py', 'exec')
    return updated


def install(path):
    source = path.read_text(encoding='utf-8')
    updated = build_source(source)
    # Hardware is never opened by this dependency check.
    import serial  # noqa: F401
    import qrcode  # noqa: F401
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = path.with_name('app.py.before_ui_qr_' + stamp)
    shutil.copy2(path, backup)
    temporary = path.with_name('app.py.ui_qr_tmp_' + stamp)
    temporary.write_text(updated, encoding='utf-8')
    temporary.chmod(path.stat().st_mode & 0o777)
    temporary.replace(path)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, default=Path('/home/photobooth/photobooth/app.py'))
    args = parser.parse_args()
    try:
        backup = install(args.app)
    except (OSError, ValueError, ImportError, SyntaxError) as error:
        raise SystemExit('적용 중단: ' + str(error))
    print('상대 연령 화면·4컷+QR 자동 인쇄 변경 준비 완료. 아직 실행 중인 프로그램에는 적용되지 않았어요.')
    print('기존 파일 백업:', backup)
    print('촬영이 끝난 뒤 sudo systemctl restart photobooth.service 로 적용하세요.')


if __name__ == '__main__':
    main()
