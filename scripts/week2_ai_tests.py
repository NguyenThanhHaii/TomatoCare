"""Offline real-artifact T09 accuracy / T10 stability. No HF download or fallback."""
import argparse
import csv
import io
import json
import platform
import importlib.metadata
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance
from tomatocare.hybrid_onnx import CLASS_NAMES, HybridOnnxPredictor, sha256_file

COUNTS = {'train': 3816, 'val': 816, 'test': 820}
EXPECTED_MODEL = '196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4'
PROTOCOL = {
    'version': 1, 'seed': 42, 'split': 'test', 'per_class': 10,
    'selection': 'sort paths; one representative per SHA-256; seeded sample per label ID',
    'accuracy_min': 0.90, 'per_class_correct_min': 8,
    'retention_min_each_variant': 0.90,
    'variants': {'brightness_080': {'factor': 0.8}, 'brightness_120': {'factor': 1.2},
                 'jpeg_q70': {'quality': 70, 'subsampling': 2, 'optimize': False, 'progressive': False}},
}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fresh(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def not_run(reason):
    return {code: {'execution_status': 'NOT_RUN', 'quality_status': 'NOT_EVALUATED', 'reason': reason}
            for code in ('T09', 'T10')}


def save_reports(out, report):
    write_json(out/'results.json', report)
    fields = ['code', 'execution_status', 'quality_status', 'total', 'correct', 'wrong', 'accuracy', 'reason']
    with (out/'case-results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, lineterminator='\n', fieldnames=fields); writer.writeheader()
        for key, case in report['cases'].items():
            writer.writerow({'code': key, **{k: case.get(k, '') for k in fields[1:]}})
    lines = ['# T09–T10: đánh giá AI thật', '', '| Mã | Thực thi | Chất lượng |', '| --- | --- | --- |']
    for key, case in report['cases'].items():
        lines.append(f"| {key} | {case['execution_status']} | {case['quality_status']} |")
    t09 = report['cases']['T09']; t10 = report['cases']['T10']
    if 'accuracy' in t09:
        lines += ['', f"T09: đúng {t09['correct']}/40; sai {t09['wrong']}/40; accuracy {t09['accuracy']:.1%}.", '',
                  '| Lớp | Đúng | Sai | Accuracy |', '| --- | --- | --- | --- |']
        for label, values in t09['per_class'].items():
            lines.append(f"| {label} | {values['correct']}/10 | {values['wrong']} | {values['accuracy']:.1%} |")
    if 'variants' in t10:
        lines += ['', '| Biến thể | Giữ nhãn | Đổi nhãn | Tỷ lệ giữ nhãn |', '| --- | --- | --- | --- |']
        for label, values in t10['variants'].items():
            lines.append(f"| {label} | {values['unchanged']}/40 | {values['changed']} | {values['retention']:.1%} |")
        with (out/'stability.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, lineterminator='\n', fieldnames=['variant', 'total', 'unchanged', 'changed', 'retention'])
            writer.writeheader()
            writer.writerows({'variant': k, **{f: v[f] for f in ('total','unchanged','changed','retention')}}
                             for k, v in t10['variants'].items())
    lines += ['', 'Giữ nguyên nhãn hoặc confidence cao không có nghĩa dự đoán đúng.',
              'Bộ 40 ảnh nhỏ không thay thế đánh giá trên toàn bộ 820 ảnh Test.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def choose(records, root, protocol):
    """Pure sampling: independent of predictions, stable under CSV row reordering."""
    rng = random.Random(protocol['seed'])
    selected, seen = [], set()
    for target, label in enumerate(CLASS_NAMES):
        pool = []
        for r in sorted(records, key=lambda r: r.path.relative_to(root).as_posix()):
            if r.split != 'test':
                raise ValueError('Selection input must contain test records only')
            if r.target != target:
                continue
            digest = sha256_file(r.path)
            if digest not in seen:
                pool.append({'image_path': r.path.relative_to(root).as_posix(), 'label_id': target,
                             'label': label, 'split': 'test', 'sha256': digest})
                seen.add(digest)
        if len(pool) < protocol['per_class']:
            raise ValueError(f'Not enough unique test images for {label}')
        selected.extend(rng.sample(pool, protocol['per_class']))
    return selected


def validated_bundle(root):
    # Reuse original metadata schema/path/split/label/leakage validation.
    from tomatocare.data import discover_bundle
    bundle = discover_bundle(root)
    if bundle.class_names != CLASS_NAMES:
        raise ValueError('labels.json order differs from Hybrid logits order')
    if {k: len(v) for k, v in bundle.records.items()} != COUNTS:
        raise ValueError('Require original dataset counts train=3816, val=816, test=820')
    declared = {}
    with (bundle.root / 'metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            declared[row['image_path']] = row['sha256'].lower()
    # Verify actual bytes across ALL splits, not only declared hashes.
    for records in bundle.records.values():
        for r in records:
            rel = r.path.relative_to(bundle.root).as_posix()
            if sha256_file(r.path) != declared.get(rel):
                raise ValueError('Image bytes do not match metadata SHA-256')
    return bundle


def prepare(data_root, out):
    bundle = validated_bundle(data_root)
    selected = choose(bundle.records['test'], bundle.root, PROTOCOL)
    # Decode selected images before freezing the cohort; no model is invoked.
    from backend.main import decode_image
    for row in selected:
        image = decode_image((bundle.root / row['image_path']).read_bytes())
        image.close()
    selection = {'protocol': PROTOCOL, 'class_names': list(CLASS_NAMES),
                 'metadata_sha256': sha256_file(bundle.root / 'metadata.csv'),
                 'labels_sha256': sha256_file(bundle.root / 'labels.json'),
                 'dataset_counts': COUNTS, 'images': selected}
    out = fresh(out)
    write_json(out / 'selection.json', selection)
    save_reports(out, {'cases': not_run('Cohort frozen; real inference not executed')})
    return selection


def verify_selection(selection, bundle):
    if selection['protocol'] != PROTOCOL or selection['class_names'] != list(CLASS_NAMES):
        raise ValueError('Protocol/classes changed; freeze a new protocol before inference')
    for name in ('metadata', 'labels'):
        ext = 'csv' if name == 'metadata' else 'json'
        if sha256_file(bundle.root / f'{name}.{ext}') != selection[f'{name}_sha256']:
            raise ValueError('Dataset metadata/labels changed since preparation')
    expected = choose(bundle.records['test'], bundle.root, selection['protocol'])
    if selection['images'] != expected or selection['dataset_counts'] != COUNTS:
        raise ValueError('Selection differs from deterministic, metadata-grounded cohort')


def variants(image):
    yield 'brightness_080', ImageEnhance.Brightness(image).enhance(0.8)
    yield 'brightness_120', ImageEnhance.Brightness(image).enhance(1.2)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=70, subsampling=2, optimize=False, progressive=False)
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        yield 'jpeg_q70', decoded.convert('RGB')


def evaluate(bundle, selection, predictor):
    from backend.main import decode_image
    rows = []
    for item in selection['images']:
        image = decode_image((bundle.root / item['image_path']).read_bytes())
        try:
            original = predictor.predict(image)
            def record(variant, result):
                # Actual shared predictor validates logits and computes softmax.
                return {**item, 'variant': variant, 'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'], 'probabilities': result['probabilities'],
                        'correct': result['predicted_class'] == item['label'],
                        'same_as_original': result['predicted_class'] == original['predicted_class'],
                        'original_prediction': original['predicted_class']}
            rows.append(record('original', original))
            for name, changed in variants(image):
                try:
                    rows.append(record(name, predictor.predict(changed)))
                finally:
                    changed.close()
        finally:
            image.close()
    originals = [r for r in rows if r['variant'] == 'original']
    correct = sum(r['correct'] for r in originals)
    per_class = {label: {'total': 10, 'correct': sum(r['correct'] for r in originals if r['label'] == label)}
                 for label in CLASS_NAMES}
    for values in per_class.values():
        values.update(wrong=10-values['correct'], accuracy=values['correct']/10)
    stability = {}
    for variant in PROTOCOL['variants']:
        group = [r for r in rows if r['variant'] == variant]
        same = sum(r['same_as_original'] for r in group)
        stability[variant] = {'total': 40, 'unchanged': same, 'changed': 40-same,
                              'retention': same/40, 'changes': [r for r in group if not r['same_as_original']]}
    cases = {
        'T09': {'execution_status': 'COMPLETED', 'quality_status': 'PASS' if correct/40 >= PROTOCOL['accuracy_min']
                and all(v['correct'] >= PROTOCOL['per_class_correct_min'] for v in per_class.values()) else 'FAIL',
                'total': 40, 'correct': correct, 'wrong': 40-correct, 'accuracy': correct/40, 'per_class': per_class},
        'T10': {'execution_status': 'COMPLETED', 'quality_status': 'PASS' if all(v['retention'] >= PROTOCOL['retention_min_each_variant']
                for v in stability.values()) else 'FAIL', 'variants': stability}}
    return cases, rows


def run(data_root, selection_file, model_dir, out, expected):
    out = fresh(out)
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': PROTOCOL,
              'cases': not_run('Prerequisites not yet validated'),
              'python_version': platform.python_version(),
              'dependencies': {k: importlib.metadata.version(k) for k in ('Pillow', 'numpy', 'onnxruntime')}}
    code = 2
    try:
        required = [Path(selection_file), Path(model_dir)/'tomatocare_hybrid.onnx', Path(model_dir)/'inference_manifest.json']
        if not Path(data_root).exists() or not all(p.is_file() for p in required):
            report['cases'] = not_run('Missing dataset, frozen selection, local ONNX or manifest; no download attempted')
            return code
        bundle = validated_bundle(data_root)
        selection = json.loads(Path(selection_file).read_text())
        verify_selection(selection, bundle)
        predictor = HybridOnnxPredictor(model_dir, expected_sha256=expected)
        # Record provenance before running; do not accept arbitrary API/mock injection.
        report.update(selection_sha256=sha256_file(selection_file), model_sha256=predictor.manifest['model_sha256'],
                      manifest_sha256=sha256_file(Path(model_dir)/'inference_manifest.json'),
                      model_version=predictor.model_version, provenance=predictor.manifest.get('provenance'),
                      selection=selection)
        cases, rows = evaluate(bundle, selection, predictor)
        report.update(cases=cases, rows=rows)
        with (out/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, lineterminator='\n', fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows({**r, 'probabilities': json.dumps(r['probabilities'])} for r in rows)
        code = 0 if all(c['quality_status'] == 'PASS' for c in cases.values()) else 1
        return code
    except Exception as error:
        report['cases'] = {key: {'execution_status': 'ERROR', 'quality_status': 'NOT_EVALUATED',
                                'reason': str(error)} for key in ('T09', 'T10')}
        return code
    finally:
        report['exit_code'] = code
        save_reports(out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan'); plan.add_argument('--out', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--data-root', required=True); prep.add_argument('--out', required=True)
    execute = sub.add_parser('run')
    for flag in ('data-root', 'selection', 'model-dir', 'out'):
        execute.add_argument('--'+flag, required=True)
    execute.add_argument('--expected-onnx-sha256', default=EXPECTED_MODEL)
    args = parser.parse_args()
    if args.command == 'plan':
        out = fresh(args.out)
        write_json(out/'protocol.json', PROTOCOL)
        save_reports(out, {'protocol': PROTOCOL, 'cases': not_run('No real dataset/model supplied; preparation only')})
        return 0
    if args.command == 'prepare':
        prepare(args.data_root, args.out); return 0
    return run(args.data_root, args.selection, args.model_dir, args.out, args.expected_onnx_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
