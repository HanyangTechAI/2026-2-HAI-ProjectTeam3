import math

import numpy as np
import torch
from PIL import ImageOps
from age.dataset import transform
from age.face import FaceAligner
from age.model import build_model, expected_age


class RealAgeEstimator:
    is_mock = False
    evaluator_id = "public_mobilenet_101"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(
            "outputs/age/public/best.pt",
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("run", {}).get("smoke_steps"):
            raise ValueError("시험용 가중치는 사용할 수 없어요.")

        self.model = build_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

        self.preprocess = transform(False)
        self.aligner = FaceAligner(
            model_root="models/insightface",
            device="cpu",
        )

    @torch.inference_mode()
    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        if min(image.size) < 16:
            raise ValueError("image_too_small")

        rgb = np.asarray(image).copy()

        # 얼굴 없음·여러 명을 검사해요.
        # 여기서는 얼굴 정렬이나 자르기를 적용하지 않아요.
        self.aligner.select(rgb, training=True)

        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        age = expected_age(self.model(batch)).item()

        if not math.isfinite(age) or not 0 <= age <= 100:
            raise ValueError("invalid_age_prediction")

        return {
            "age": age,
            "gender": "unknown",
            "is_mock": False,
        }
