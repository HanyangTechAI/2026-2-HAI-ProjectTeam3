# 포토부스 작업 상태 — 2026-11-24

## 확인 완료
- 실제 웹캠 촬영 사진으로 나이 추정 성공. 모의 추정이 아님.
- FADING으로 실제 촬영 사진의 4컷 생성 성공.
- 로딩·생성·합성을 포함한 한 번의 측정값: 100.8초.
- 라즈베리파이 웹 촬영·저장 버튼과 좌우반전 코드 추가.
- 연결 시험 프로그램에서 라즈베리파이→맥→HAI 통신 성공.

## 코드
- raspi/photobooth_app.py: 라즈베리파이에서 가져온 촬영·인쇄 코드.
- scripts/fading_infer.py: FADING 실행 도구.
- app/fading_booth_api.py: 실제 나이 추정·FADING 4컷 요청 처리 코드.

## 미완료 및 다음 작업
- 실제 처리 프로그램 전환 후 통신 타임아웃 발생. 실행 상태와 연결 재확인.
- 실제 4컷 요청·응답 시험.
- 웹 촬영 버튼에 자동 전송·처리 상태·결과 표시 연결.
- 결과 다운로드 QR 및 생성된 4컷 인쇄 연결.
- 현재 촬영부터 QR까지 자동 처리가 완료된 상태는 아님.

## 실행 구성
- 라즈베리파이: /home/photobooth/photobooth/app.py
- 라즈베리파이 실행 환경: /home/photobooth/photobooth/env/
- 실행 서비스: photobooth.service
- HAI의 8000번을 맥으로 포트 전달.
- 맥의 SSH 중계로 라즈베리파이 18000번에서 HAI에 접근.
- 맥과 두 전달 연결을 유지해야 함.

## 복원 및 보관
- FADING: ~/.cache/fading-src, ~/.cache/fading-models, ~/.cache/fading-env
- 나이 추정 가중치는 Git LFS로 복원.
- 얼굴 검출 관련 파일은 python -m age.setup_models로 복원.
- 가중치·가상환경·사진·생성 결과는 이번 코드 저장에 포함하지 않음.
- HAI 사진·결과는 아직 자동 삭제되지 않음.
- 목표 연령은 생성 조건이며 실제 미래 얼굴이나 나이 정답이 아님.
