"""Preparation-only unit checks: no mock classifier, no A09 robustness claim."""
import json

import numpy as np
import pytest

from scripts.week2_adversarial import SETTINGS, bounded_candidate, run


def test_perturbation_stays_in_original_linf_ball():
    original=np.array([[[0,128,255],[255,0,128]]],dtype=np.uint8)
    best=original.copy()
    for _ in range(20):
        best=bounded_candidate(original,best,np.ones_like(best,dtype=np.int16))
    assert best.dtype==np.uint8 and best.shape==original.shape
    assert np.abs(best.astype(int)-original.astype(int)).max()<=8
    for _ in range(20):
        best=bounded_candidate(original,best,-np.ones_like(best,dtype=np.int16))
    assert np.abs(best.astype(int)-original.astype(int)).max()<=8
    assert SETTINGS['max_images']*(1+SETTINGS['queries_per_image'])==36


def test_a09_missing_artifacts_never_claims_blocked(tmp_path):
    out=tmp_path/'results'
    assert run(tmp_path/'missing-data',tmp_path/'selection.json',tmp_path/'model',out)==2
    report=json.loads((out/'a09-results.json').read_text())
    assert report['assessment']=='NOT_RUN' and report['total_queries']==0
    assert report['images']==[]
    assert not list(out.glob('*.png'))
    with pytest.raises(FileExistsError):
        run(tmp_path/'data',tmp_path/'selection.json',tmp_path/'model',out)
