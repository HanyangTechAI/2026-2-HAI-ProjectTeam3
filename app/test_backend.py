import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_generate_qr():
    response = client.get("/api/v1/generate-qr?url=https://trycloudflare.com/photos/123")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    img = Image.open(io.BytesIO(response.content))
    assert img.format == "PNG"
    print("\n[PASS] QR Code Generation Test Completed!")
