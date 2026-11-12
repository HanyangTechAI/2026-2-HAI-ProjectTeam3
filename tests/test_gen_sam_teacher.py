import unittest

import torch
from PIL import Image

from gen.sam_teacher import SamTeacher, to_input


class FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, randomize_noise=True, resize=True):
        self.calls.append(dict(shape=tuple(x.shape), randomize_noise=randomize_noise, resize=resize))
        return x[:, :3]


class SamTeacherContracts(unittest.TestCase):
    def test_input_has_image_and_age_channel(self):
        tensor = to_input(Image.new('RGB', (512, 512), (255, 0, 128)), 65)
        self.assertEqual(tuple(tensor.shape), (4, 256, 256))
        self.assertAlmostEqual(float(tensor[0].mean()), 1.0, places=5)
        self.assertAlmostEqual(float(tensor[1].mean()), -1.0, places=5)
        self.assertTrue(torch.allclose(tensor[3], torch.full((256, 256), .65)))
        for age in (-1, 101):
            with self.subTest(age=age), self.assertRaises(ValueError):
                to_input(Image.new('RGB', (8, 8)), age)

    def test_age_batches_and_disables_noise(self):
        net = FakeNet()
        teacher = SamTeacher(None, None, device='cpu', net=net)
        outputs = teacher.age([Image.new('RGB', (512, 512), (10, 200, 30))] * 2, [25, 75])
        self.assertEqual(net.calls, [dict(shape=(2, 4, 256, 256), randomize_noise=False, resize=False)])
        self.assertEqual([image.size for image in outputs], [(512, 512)] * 2)
        self.assertEqual(outputs[0].getpixel((5, 5)), (10, 200, 30))
        with self.assertRaises(ValueError):
            teacher.age([Image.new('RGB', (8, 8))], [20, 30])


if __name__ == '__main__':
    unittest.main()
