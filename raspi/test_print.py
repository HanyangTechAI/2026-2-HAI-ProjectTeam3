import serial
import time
from PIL import Image, ImageDraw, ImageFont

def wake_up_and_clear_printer(ser):
    # 프린터 버퍼 깨우기 및 초기화 시퀀스
    ser.write(b'\x00\x00\x00')
    time.sleep(0.05)
    ser.write(b'\x1b\x40')
    time.sleep(0.05)
    ser.write(b'\x1b\x3d\x01')
    time.sleep(0.1)

def print_test_pattern():
    PRINTER_WIDTH = 384
    img_height = 100

    # 1비트 흰 바탕 이미지 생성 및 테스트 텍스트/선 그리기
    img = Image.new('1', (PRINTER_WIDTH, img_height), 1)
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([10, 10, PRINTER_WIDTH - 10, img_height - 10], outline=0, width=3)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((30, 35), "TEST PRINT SUCCESS!", fill=0, font=font)

    # ESC/POS 래스터 비트맵 커맨드 (0x1D 0x76 0x30 0x00)
    img_byte_width = PRINTER_WIDTH // 8
    header = bytearray([0x1D, 0x76, 0x30, 0x00, img_byte_width & 0xFF, (img_byte_width >> 8) & 0xFF, img_height & 0xFF, (img_height >> 8) & 0xFF])

    bitmap_data = bytearray()
    for py in range(img_height):
        for bx in range(img_byte_width):
            byte_val = 0
            for bit in range(8):
                px = bx * 8 + bit
                if px < PRINTER_WIDTH:
                    if img.getpixel((px, py)) == 0:
                        byte_val |= (1 << (7 - bit))
            bitmap_data.append(byte_val)

    print("프린터 포트 연결 중...")
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
    
    print("Wake-up 신호 전송...")
    wake_up_and_clear_printer(ser)
    
    print("비트맵 이미지 데이터 전송...")
    ser.write(header + bitmap_data)
    ser.write(b'\n\n\n\n')
    ser.flush()
    ser.close()
    print("전송 완료!")

if __name__ == '__main__':
    print_test_pattern()
