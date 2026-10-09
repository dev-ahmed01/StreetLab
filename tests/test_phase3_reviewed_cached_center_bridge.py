"""SHA-checked case-level detector replay never changes FLUID scores."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

import streetlab_phase3.video.reviewed_cached_center_bridge as bridge


def sha(data):return hashlib.sha256(data).hexdigest()

@pytest.fixture
def evidence(tmp_path,monkeypatch):
    review=tmp_path/'pack'
    review.mkdir()
    (review/'manifest.json').write_bytes(b'blind manifest fixture')
    manifest={
        'source_bundle_manifest_sha256':'', 'case_count':3,
        'cases':[
            {'case_id':'C1','source_frame':7,'crop_origin':[0,0],
             'crop_size':480,'anchor_in_crop':[150,150]},
            {'case_id':'C2','source_frame':7,'crop_origin':[0,0],
             'crop_size':480,'anchor_in_crop':[250,250]},
            {'case_id':'C3','source_frame':7,'crop_origin':[0,0],
             'crop_size':480,'anchor_in_crop':[350,350]},
        ],
    }
    for n in range(4,16):
        manifest['cases'].append({'case_id':f'C{n}',
            'source_frame':7,'crop_origin':[0,0],
            'crop_size':480,'anchor_in_crop':[350,350]})
    manifest['case_count']=len(manifest['cases'])
    archive=tmp_path/'evidence.zip'
    files={}
    for backend,pred in (('openvino',[(150,150,'BUS'),(250,250,'CAR')]),
                         ('pytorch',[(150,150,'BUS')])):
        data=io.StringIO()
        w=csv.writer(data)
        w.writerow(['frame','x_px','y_px','vehicle_class','confidence'])
        for x,y,c in pred:w.writerow([7,x,y,c,.7])
        member=f'audits/{backend}/sliced_detections.csv'
        files[member]=data.getvalue().encode()
        report={'sample_frames':[7], 'model_sha256':'a'*64, 'runtime_model_sha256':'b'*64, 'sliced':{
            'predicted_points':len(pred),'matched_points':1,'truth_points':1,
            'per_class':{c:{'predicted':sum(cc==c for _,_,cc in pred)}
                         for c in bridge.CANONICAL}}}
        files[f'audits/{backend}/report.json']=json.dumps(report).encode()
    src={'status':'UNPROMOTED_W04_SELF_CONTAINED_RESEARCH_INPUT',
         'frames':[7], 'original_model_sha256':'a'*64,'openvino_export_sha256':'b'*64,
         'files':{key:sha(value) for key,value in files.items()}}
    manifest_bytes=json.dumps(src).encode()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('manifest.json',manifest_bytes)
        for key,val in files.items():z.writestr(key,val)
    manifest['source_bundle_manifest_sha256']=sha(manifest_bytes)
    monkeypatch.setattr(bridge,'validate_pack',lambda _:manifest)
    consensus=tmp_path/'consensus.json'
    consensus.write_text(json.dumps({
        'status':bridge.CONSENSUS_STATUS,'eligible_for_promotion':False,
        'case_count':15,'review_manifest_sha256':sha((review/'manifest.json').read_bytes()),
        'cases':[
            {'case_id':'C1','decision':'AGREED_PRESENT','physical_class':'BUS',
             'consensus_bbox_crop':[125,125,175,175]},
            {'case_id':'C2','decision':'AGREED_PRESENT','physical_class':'AUTO_RICKSHAW',
             'consensus_bbox_crop':[225,225,275,275]},
            {'case_id':'C3','decision':'UNRESOLVED','physical_class':None},
            *[{'case_id':f'C{n}','decision':'AGREED_PRESENT',
               'physical_class':'MOTORCYCLE',
               'consensus_bbox_crop':[325,325,375,375]}
              for n in range(4,15)],
            {'case_id':'C15','decision':'UNRESOLVED','physical_class':None},
        ],
    }))
    return archive,review,consensus,manifest


def test_two_backend_case_level_comparison(evidence,monkeypatch):
    bundle,review,consensus,pack=evidence
    result=bridge.compare_reviewed_centers(bundle=bundle,review_dir=review,consensus_file=consensus)
    ov=result['case_level_center_coverage_by_backend']['openvino']
    pt=result['case_level_center_coverage_by_backend']['pytorch']
    assert ov['agreed_present_cases']==13
    assert ov['localized_any_class_cases']==2
    assert ov['localized_compatible_class_cases']==1
    assert ov['physical_class_not_in_detector_ontology']==1
    assert pt['localized_any_class_cases']==1
    assert result['physical_precision'] is None
    assert sum(x['decision']=='UNRESOLVED' for x in result['cases'])==2


def test_rejects_tampered_consensus(evidence):
    bundle,review,consensus,pack=evidence
    c=json.loads(consensus.read_text())
    c['review_manifest_sha256']='corrupt'
    consensus.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='provenance mismatch'):
        bridge.compare_reviewed_centers(bundle=bundle,review_dir=review,consensus_file=consensus)


def test_rejects_wrong_evidence_bundle(evidence):
    bundle,review,consensus,pack=evidence
    pack['source_bundle_manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='differs from reviewer pack'):
        bridge.compare_reviewed_centers(bundle=bundle,review_dir=review,consensus_file=consensus)


def test_real_w04_review_evidence_optional():
    bundle=Path('/mnt/data/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip')
    pack=Path('/mnt/data/W04_BLIND_REVIEW_15_02')
    consensus=Path('/mnt/data/W04_manual_dual_consensus_20261009_01.json')
    if not (bundle.exists() and pack.exists() and consensus.exists()):
        pytest.skip('Private W04 evidence not available in public CI')
    r=bridge.compare_reviewed_centers(bundle=bundle,review_dir=pack,
                                      consensus_file=consensus)
    assert r['status']==bridge.STATUS
    assert r['case_level_center_coverage_by_backend']['openvino']['localized_compatible_class_cases']==12
    assert r['case_level_center_coverage_by_backend']['pytorch']['localized_compatible_class_cases']==11
    assert sum(case['decision']=='UNRESOLVED' for case in r['cases'])==2
    assert r['physical_precision'] is None