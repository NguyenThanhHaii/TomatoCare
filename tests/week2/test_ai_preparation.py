"""Synthetic selection/transform integrity only. Never runs/replaces real AI T09/T10."""
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.week2_ai_tests import COUNTS, PROTOCOL, choose, fresh, run, variants, verify_selection
from tomatocare.hybrid_onnx import CLASS_NAMES, sha256_file


def test_seeded_test_only_cohort_is_stable(tmp_path):
    records = []
    for target in range(4):
        for index in range(12):
            path = tmp_path / f'{target}-{index}.png'
            Image.new('RGB', (8, 8), (target*40, index*10, 30)).save(path)
            records.append(SimpleNamespace(path=path, target=target, split='test'))
    a = choose(records, tmp_path, PROTOCOL)
    assert a == choose(list(reversed(records)), tmp_path, PROTOCOL)
    assert len(a) == len({r['sha256'] for r in a}) == 40
    assert all(sum(r['label_id'] == target for r in a) == 10 for target in range(4))
    assert all(r['split'] == 'test' for r in a)
    (tmp_path/'metadata.csv').write_text('SYNTHETIC sampling unit fixture only')
    (tmp_path/'labels.json').write_text(json.dumps(list(CLASS_NAMES)))
    selection = {'protocol': PROTOCOL, 'class_names': list(CLASS_NAMES), 'images': a,
                 'dataset_counts': COUNTS,
                 'metadata_sha256': sha256_file(tmp_path/'metadata.csv'),
                 'labels_sha256': sha256_file(tmp_path/'labels.json')}
    bundle = SimpleNamespace(root=tmp_path, records={'test': records})
    verify_selection(selection, bundle)
    selection['images'][0]['label'] = 'WRONG FIXTURE LABEL'
    with pytest.raises(ValueError, match='Selection differs'):
        verify_selection(selection, bundle)
    records[0].split = 'val'
    with pytest.raises(ValueError, match='test records only'):
        choose(records, tmp_path, PROTOCOL)


def test_transform_definition_without_ai():
    image = Image.new('RGB', (17, 19), (100, 100, 100))
    changed = dict(variants(image))
    try:
        assert changed['brightness_080'].getpixel((0, 0)) == (80, 80, 80)
        assert changed['brightness_120'].getpixel((0, 0)) == (120, 120, 120)
        assert changed['jpeg_q70'].mode == 'RGB'
        assert all(v.size == image.size for v in changed.values())
        assert image.getpixel((0, 0)) == (100, 100, 100)
    finally:
        for v in changed.values(): v.close()
        image.close()


def test_missing_prerequisites_remain_not_run(tmp_path):
    out = tmp_path/'report'
    assert run(tmp_path/'absent-data', tmp_path/'absent-selection', tmp_path/'absent-model', out, '0'*64) == 2
    report = json.loads((out/'results.json').read_text())
    assert all(c['execution_status'] == 'NOT_RUN' and c['quality_status'] == 'NOT_EVALUATED'
               for c in report['cases'].values())
    assert not (out/'predictions.csv').exists()
    with pytest.raises(FileExistsError): fresh(out)
