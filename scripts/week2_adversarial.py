"""A09: bounded score-guided black-box search against a LOCAL real Hybrid ONNX.

No remote API, HF download, training, weight modification, or synthetic model.
Only a four-image subset of the original frozen Test cohort is read.
"""
import argparse
import csv
import hashlib
import json
import platform
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.week2_ai_tests import (EXPECTED_MODEL, fresh, validated_bundle,
                                    verify_selection, write_json)
from tomatocare.hybrid_onnx import CLASS_NAMES, HybridOnnxPredictor, sha256_file

SETTINGS = {'seed': 2026, 'epsilon_uint8': 8, 'step_uint8': 2, 'queries_per_image': 8,
            'max_images': 4, 'max_total_queries': 36, 'max_pixels_per_image': 1_000_000,
            'method': 'score-guided random low-frequency sign search (untargeted)',
            'success': 'original correct AND perturbed prediction != ground truth; Linf RGB <= 8/255'}


def bounded_candidate(original, best, direction):
    """uint8 RGB domain only; preserve image size and clamp against original."""
    base=original.astype(np.int16)
    candidate=best.astype(np.int16)+SETTINGS['step_uint8']*direction.astype(np.int16)
    candidate=np.maximum(base-SETTINGS['epsilon_uint8'],np.minimum(base+SETTINGS['epsilon_uint8'],candidate))
    return np.clip(candidate,0,255).astype(np.uint8)


def export(out,report):
    write_json(out/'a09-results.json',report)
    fields=['image_path','label','original_prediction','final_prediction','queries',
            'max_rgb_delta','success','eligible','reason','adversarial_png_sha256']
    with (out/'a09-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows({k:row.get(k,'') for k in fields} for row in report.get('images',[]))
    if not report.get('images'):
        # Explicit NOT_RUN/ERROR row rather than a misleading empty success report.
        with (out/'a09-case.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['code','assessment','execution_status','reason'],lineterminator='\n')
            writer.writeheader();writer.writerow({k:report.get(k,'') for k in writer.fieldnames})
    text=['# A09 — Hybrid ONNX thật', '', f"Đánh giá: **{report['assessment']}**; thực thi: {report['execution_status']}."]
    if report['execution_status']=='COMPLETED':
        text += ['', f"Số query: {report['total_queries']} / 36; số ảnh tấn công thành công: {report['successful_images']}."]
    else:
        text += ['', 'Chưa có kết quả AI hoàn tất. Không suy diễn thành công/thất bại của tấn công.']
    text += ['', 'Không tìm được ảnh đối kháng với ngân sách nhỏ KHÔNG chứng minh hệ thống chống được mọi tấn công.',
             'Không thay thế hoặc ghi lại kết quả T09–T10. Chi tiết xem JSON/CSV; ảnh thành công chỉ lưu ở output riêng.']
    (out/'a09-summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def run(data_root,selection_path,model_dir,out_path,expected=EXPECTED_MODEL):
    out=fresh(out_path)
    report={'python_version':platform.python_version(),'generated_at_utc':datetime.now(timezone.utc).isoformat(),'code':'A09','assessment':'NOT_RUN','execution_status':'NOT_RUN',
            'settings':SETTINGS,'total_queries':0,'images':[],
            'reason':'Real local model, manifest, frozen Test selection and original data are required'}
    code=2
    stage='prerequisites'
    try:
        required=[Path(selection_path),Path(model_dir)/'tomatocare_hybrid.onnx',Path(model_dir)/'inference_manifest.json']
        if not Path(data_root).exists() or not all(p.is_file() for p in required):
            return code
        stage='dataset validation'
        bundle=validated_bundle(data_root)
        selection=json.loads(Path(selection_path).read_text())
        stage='frozen cohort verification'
        verify_selection(selection,bundle)
        selected=[next(r for r in selection['images'] if r['label']==label) for label in CLASS_NAMES]
        stage='trusted ONNX session initialization'
        predictor=HybridOnnxPredictor(model_dir,expected_sha256=expected)
        report.update(dependencies={name:importlib.metadata.version(name) for name in ('Pillow','numpy','onnxruntime')},
                      execution_providers=predictor.session.get_providers(),
                      selection_sha256=sha256_file(selection_path),onnx_sha256=predictor.manifest['model_sha256'],
                      manifest_sha256=sha256_file(Path(model_dir)/'inference_manifest.json'),
                      model_version=predictor.model_version)
        rng=np.random.default_rng(SETTINGS['seed'])
        from backend.main import decode_image, MAX_BYTES
        stage='bounded search'
        for item in selected:
            path=bundle.root/item['image_path']
            if path.stat().st_size>MAX_BYTES:
                raise ValueError('Image exceeds A09 byte budget')
            payload=path.read_bytes()
            # Check dimensions from the header BEFORE decoding/allocating pixels.
            import io
            with Image.open(io.BytesIO(payload)) as header:
                if header.width*header.height>SETTINGS['max_pixels_per_image']:
                    raise ValueError('Image exceeds A09 pixel budget')
            image=decode_image(payload)
            try:
                if image.width*image.height>SETTINGS['max_pixels_per_image']:
                    raise ValueError('Image exceeds conservative A09 pixel budget; no automatic resizing')
                original=np.asarray(image).copy()
                baseline=predictor.predict(image);report['total_queries']+=1
                row={'image_path':item['image_path'],'source_sha256':item['sha256'],'label':item['label'],
                     'original_prediction':baseline['predicted_class'],'original_confidence':baseline['confidence'],
                     'final_prediction':baseline['predicted_class'],'final_confidence':baseline['confidence'],'queries':1,'max_rgb_delta':0,
                     'success':False,'eligible':baseline['predicted_class']==item['label']}
                if not row['eligible']:
                    row['reason']='Originally wrong: not an adversarial success; perturbations not queried'
                    report['images'].append(row);continue
                best=original.copy();score=baseline['probabilities'][item['label']]
                row['trace']=[]
                for _ in range(SETTINGS['queries_per_image']):
                    if report['total_queries']>=SETTINGS['max_total_queries']:
                        break
                    # Coarse signs, expanded in memory, no extra model queries/gradient.
                    small=rng.choice(np.array([-1,1],dtype=np.int16),size=(8,8,3))
                    direction=np.repeat(np.repeat(small,(image.height+7)//8,axis=0),(image.width+7)//8,axis=1)[:image.height,:image.width]
                    candidate=bounded_candidate(original,best,direction)
                    delta=int(np.abs(candidate.astype(np.int16)-original.astype(np.int16)).max())
                    assert delta<=SETTINGS['epsilon_uint8']
                    with Image.fromarray(candidate) as changed:
                        prediction=predictor.predict(changed)
                    row['queries']+=1;report['total_queries']+=1
                    row['trace'].append({'prediction':prediction['predicted_class'],
                                         'true_class_probability':prediction['probabilities'][item['label']],
                                         'max_rgb_delta':delta})
                    success=prediction['predicted_class']!=item['label']
                    if success or prediction['probabilities'][item['label']]<score:
                        best=candidate;score=prediction['probabilities'][item['label']]
                        row.update(final_prediction=prediction['predicted_class'],
                                   final_confidence=prediction['confidence'],max_rgb_delta=delta)
                    if success:
                        row['success']=True;row['reason']='Originally correct; bounded perturbation caused wrong label'
                        filename=f'adversarial-{len(report["images"]):02d}.png'
                        with Image.fromarray(candidate) as changed: changed.save(out/filename,format='PNG')
                        row['adversarial_png']=filename
                        row['adversarial_png_sha256']=sha256_file(out/filename)
                        # Lossless export is verified to preserve the exact tested bytes.
                        with Image.open(out/filename) as saved:
                            assert hashlib.sha256(saved.tobytes()).digest()==hashlib.sha256(candidate.tobytes()).digest()
                        break
                if not row['success']:
                    row['reason']='No label flip within this bounded search; robustness not established'
                report['images'].append(row)
            finally:image.close()
        successes=sum(r['success'] for r in report['images'])
        eligible=sum(r['eligible'] for r in report['images'])
        report.update(execution_status='COMPLETED',assessment='WEAKNESS_FOUND' if successes else 'INCONCLUSIVE',
                      successful_images=successes,eligible_images=eligible,
                      empirical_success_rate=successes/eligible if eligible else None,
                      reason='Limited query-budget experiment, not a robustness certification')
        code=0
        return code
    except Exception as error:
        report.update(execution_status='ERROR',assessment='INCONCLUSIVE',
                      reason='Failure at '+stage+': '+type(error).__name__)
        return code
    finally:
        report['exit_code']=code
        export(out,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('data-root','selection','model-dir','out'):
        parser.add_argument('--'+flag,required=True)
    parser.add_argument('--expected-onnx-sha256',default=EXPECTED_MODEL)
    args=parser.parse_args()
    return run(args.data_root,args.selection,args.model_dir,args.out,args.expected_onnx_sha256)


if __name__=='__main__':
    raise SystemExit(main())
