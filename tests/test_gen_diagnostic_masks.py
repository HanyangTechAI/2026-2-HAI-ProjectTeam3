import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from age.common import sha256
from gen.prepare_diagnostic_masks import polygon_mask, prepare


class DiagnosticMaskTests(unittest.TestCase):
    def test_polygon_bounds_and_inward_feather_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'source.png'
            Image.new('RGB', (32, 32), 'blue').save(image)
            sample = dict(sample_id='test', input_path=str(image), sha256=sha256(image))
            annotation = dict(source_size=[32, 32], editable=[[[4, 4], [27, 4], [27, 27], [4, 27]]],
                              protected=[[[8, 8], [12, 8], [12, 12]]], negative_prompt='test', notes='synthetic')
            recipe = root / 'annotations.json'
            recipe.write_text(json.dumps(dict(author='assistant_visual_diagnostic', human_verified=False,
                                              samples={'test': annotation})))
            source_manifest = root / 'val.json'
            source_manifest.write_text('{}')
            with patch('gen.prepare_diagnostic_masks.load_manifest', return_value={'samples': [sample]}):
                result = prepare(source_manifest, recipe, root / 'output')
                with self.assertRaises(FileExistsError):
                    prepare(source_manifest, recipe, root / 'output')
            self.assertFalse(result['human_verified'])
            asset = result['samples']['test']['source_edit_mask']
            self.assertEqual(sha256(asset['path']), asset['sha256'])
            with Image.open(asset['path']) as mask:
                array = np.asarray(mask)
                self.assertEqual(array[:4].sum(), 0)
                self.assertEqual(array[:, :4].sum(), 0)
                self.assertGreater(array[16, 16], array[4, 4])
            self.assertEqual(sha256(image), sample['sha256'])

    def test_invalid_coordinates_are_rejected(self):
        for polygon in ([[[1, 1], [2, 2]]], [[[0, 0], [32, 0], [0, 2]]],
                        [[[0, 0], [2.5, 0], [0, 2]]]):
            with self.assertRaises(ValueError):
                polygon_mask((32, 32), polygon)


if __name__ == '__main__':
    unittest.main()
