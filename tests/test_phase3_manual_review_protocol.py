"""Two independent human reviewers, tested with SYNTHETIC human answers only."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from streetlab_phase3.video.manual_review_protocol import (
    generate_review_pack, validate_pack, adjudicate_pair,
    evaluate_adjudicated_candidates, _hash)

BUNDLE=Path('/mnt/data/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip')

@pytest.fixture(scope='module')
def pack(tmp_path_factory):
    if not BUNDLE.is_file():
        pytest.skip('Optional original W04 source bundle unavailable')
    root=tmp_path_factory.mktemp('w04review')
    folder=root/'cases'
    r=generate_review_pack(BUNDLE,folder)
    return folder,r


def fill(pack, dest, reviewer_id, *, statuses=None):
    statuses=statuses or {}
    with (pack/'reviewer_A_TEMPLATE.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    cases=json.loads((pack/'manifest.json').read_text())['cases']
    for i,(row,case) in enumerate(zip(rows,cases)):
        state,klass=statuses.get(i,('ABSENT',''))
        row['reviewer_id']=reviewer_id;row['presence']=state
        row['physical_class']=klass
        if state=='PRESENT':
            cx,cy=case['anchor_in_crop']
            size=case['crop_size']
            row.update(x1=str(max(0,int(cx)-24)),y1=str(max(0,int(cy)-24)),
                x2=str(min(size,int(cx)+24)),y2=str(min(size,int(cy)+24)))
        else:
            row.update(x1='',y1='',x2='',y2='')
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return dest


def test_blinded_pack_not_auto_labeled_and_source_images_verified(pack):
    folder,manifest=pack
    assert manifest['status']=='W04_BLIND_CASE_REVIEW_NOT_OFFICIAL_GROUND_TRUTH'
    assert manifest['case_count']==15
    assert len(list((folder/'images').glob('*.jpg')))==15
    assert len({c['case_id'] for c in manifest['cases']})==15
    safe=json.dumps(manifest)
    for bad in ('nearest_raw_fluid_labels','raw_fluid_type','moped','BUS', 'selected_class'):
        assert bad not in safe
    assert len(manifest['source_bundle_manifest_sha256'])==64
    assert validate_pack(folder)==manifest


def test_no_automatic_consensus_or_same_reviewer(pack,tmp_path):
    folder,_=pack
    with pytest.raises(ValueError,match='reviewer identifier|Incomplete/invalid review'):
        adjudicate_pair(folder,folder/'reviewer_A_TEMPLATE.csv',
                        folder/'reviewer_B_TEMPLATE.csv',tmp_path/'out.json')
    a=fill(folder,tmp_path/'a.csv','SAME')
    b=fill(folder,tmp_path/'b.csv','SAME')
    with pytest.raises(ValueError,match='two different people'):
        adjudicate_pair(folder,a,b,tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()


def test_two_reviewer_class_disagreement_and_unknown_never_consensus(pack,tmp_path):
    folder,_=pack
    a=fill(folder,tmp_path/'a.csv','independent_A',statuses={0:('PRESENT','BUS'),1:('PRESENT','CAR'),2:('UNCERTAIN','')})
    b=fill(folder,tmp_path/'b.csv','independent_B',statuses={0:('PRESENT','BUS'),1:('PRESENT','MOTORCYCLE'),2:('PRESENT','UNKNOWN')})
    c=adjudicate_pair(folder,a,b,tmp_path/'agreed.json')
    assert c['decisions']=={'AGREED_PRESENT':1,'UNRESOLVED':2,'AGREED_ABSENT':12}
    assert c['cases'][0]['decision']=='AGREED_PRESENT'
    assert c['cases'][0]['physical_class']=='BUS'
    assert c['cases'][1]['decision']=='UNRESOLVED'
    assert c['cases'][2]['decision']=='UNRESOLVED'
    assert c['physical_precision'] is None
    assert c['full_frame_recall'] is None
    with pytest.raises(FileExistsError):adjudicate_pair(folder,a,b,tmp_path/'agreed.json')


def test_disagree_on_bbox_is_unresolved(pack,tmp_path):
    folder,_=pack
    a=fill(folder,tmp_path/'a.csv','A',statuses={0:('PRESENT','MOTORCYCLE')})
    b=fill(folder,tmp_path/'b.csv','B',statuses={0:('PRESENT','MOTORCYCLE')})
    rows=list(csv.DictReader(b.open(newline='',encoding='utf-8')))
    anchor=json.loads((folder/'manifest.json').read_text())['cases'][0]['anchor_in_crop']
    # Reviewer B draws a valid small box at anchor, low overlap with A.
    cx,cy=anchor
    rows[0].update(x1=str(int(cx)-1),y1=str(int(cy)-1),
                   x2=str(int(cx)+1),y2=str(int(cy)+1))
    with b.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    c=adjudicate_pair(folder,a,b,tmp_path/'c.json')
    assert c['cases'][0]['decision']=='UNRESOLVED'
    assert c['cases'][0]['reviewer_bbox_iou']<.5


def test_reject_corrupted_image_and_bbox_out_of_bounds(pack,tmp_path):
    folder,_=pack
    cloned=tmp_path/'copied';shutil.copytree(folder,cloned)
    with (cloned/'images'/'case_01.jpg').open('ab') as f:f.write(b'spoof')
    with pytest.raises(ValueError,match='Tampered'):validate_pack(cloned)
    a=fill(folder,tmp_path/'a.csv','A',statuses={0:('PRESENT','CAR')})
    b=fill(folder,tmp_path/'b.csv','B',statuses={0:('PRESENT','CAR')})
    rows=list(csv.DictReader(a.open(newline='',encoding='utf-8')))
    rows[0]['x2']='999999'
    with a.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='outside left crop'):
        adjudicate_pair(folder,a,b,tmp_path/'c.json')


def test_manual_box_check_against_single_candidate_without_any_fluid_relabel(pack,tmp_path):
    folder,manifest=pack
    a=fill(folder,tmp_path/'a.csv','person_a',statuses={0:('PRESENT','MOTORCYCLE')})
    b=fill(folder,tmp_path/'b.csv','person_b',statuses={0:('PRESENT','MOTORCYCLE')})
    consensus_path=tmp_path/'consensus.json'
    adjudicate_pair(folder,a,b,consensus_path)
    # Synthetic geometry from a test double; NOT W04 detector inference.
    lab=tmp_path/'lab';(lab/'candidate_boxes').mkdir(parents=True)
    reviewed=manifest['cases'][0];ox,oy=reviewed['crop_origin'];x,y=reviewed['anchor_in_crop']
    candidate=lab/'candidate_boxes'/'fixture.csv'
    with candidate.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(('frame','x1','y1','x2','y2','vehicle_class','confidence'))
        w.writerow((reviewed['source_frame'],ox+x-8,oy+y-8,ox+x+8,oy+y+8,'MOTORCYCLE',.87))
    report={'status':'EXPERIMENTAL_PRETRACK_BOX_MATRIX_W04_TUNING_ONLY',
            'eligible_for_promotion':False,'source_provenance':{
                'manifest_sha256':manifest['source_bundle_manifest_sha256']},
            'frames':sorted({c['source_frame'] for c in manifest['cases']}),
            'candidate_count':1,'candidates':[{'name':'fixture'}],
            'candidate_box_csv_sha256':{'fixture':_hash(candidate.read_bytes())}}
    (lab/'matrix_report.json').write_text(json.dumps(report))
    outcome=evaluate_adjudicated_candidates(pack_dir=folder,consensus_file=consensus_path,
        box_lab_dir=lab,output=tmp_path/'out.json')
    assert outcome['candidate_count']==1
    counts=outcome['candidates'][0]['reviewed_case_counts']
    assert counts['correct_class_center_inside_manual_box']==1
    assert counts['agreed_present_cases']==1
    assert counts['agreed_absent_cases']==14
    assert outcome['physical_precision'] is None
    assert outcome['full_frame_recall'] is None
    with pytest.raises(FileExistsError):
        evaluate_adjudicated_candidates(pack_dir=folder,consensus_file=consensus_path,
            box_lab_dir=lab,output=tmp_path/'out.json')
    with candidate.open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='checksum changed'):
        evaluate_adjudicated_candidates(pack_dir=folder,consensus_file=consensus_path,
            box_lab_dir=lab,output=tmp_path/'other.json')

@pytest.fixture
def synthetic_pack(tmp_path,monkeypatch):
    """CI fixture works without the user's private W04 archive."""
    import zipfile,io
    from PIL import Image
    import streetlab_phase3.video.manual_review_protocol as module
    source=tmp_path/'fake_bundle.zip'
    pic=Image.new('RGB',(3840,2160),(170,170,170))
    bio=io.BytesIO();pic.save(bio,format='JPEG',quality=80)
    cases={'status':'W04_RAW_FLUID_LABEL_ONTOLOGY_REVIEW_NOT_RELABELED',
           'eligible_for_promotion':False,'case_count':1,'cases':[
               {'source_frame':100,'selected_x_px':250.,'selected_y_px':300.,
                'selected_class':'BUS','nearest_raw_fluid_labels':[
                    {'raw_fluid_type':'car'}]}]}
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('frames/frame_000100.jpg',bio.getvalue())
        z.writestr('previous/W04_raw_fluid_label_nearest15_01.json',json.dumps(cases))
    monkeypatch.setattr(module,'bundle_data',lambda _:(
        [100],[],{'manifest_sha256':'f'*64}))
    dest=tmp_path/'pack';module.generate_review_pack(source,dest,case_limit=1)
    return dest


def test_ci_synthetic_blind_review_and_html(synthetic_pack):
    folder=synthetic_pack
    r=validate_pack(folder)
    assert len(r['cases'])==1
    ui=(folder/'reviewer_ui.html').read_text()
    assert 'get(\'download\')' in ui
    assert 'case_01.jpg' in ui
    assert 'nearest_raw_fluid_labels' not in ui
    assert 'selected_class' not in ui
    assert 'source_frame_jpeg_sha256' in str(r)


def test_ci_synthetic_consensus_and_nonpromotion(synthetic_pack,tmp_path):
    a=fill(synthetic_pack,tmp_path/'a.csv','ann_A',statuses={0:('PRESENT','BUS')})
    b=fill(synthetic_pack,tmp_path/'b.csv','ann_B',statuses={0:('PRESENT','BUS')})
    r=adjudicate_pair(synthetic_pack,a,b,tmp_path/'agreed.json')
    assert r['decisions']=={'AGREED_PRESENT':1}
    assert r['is_original_FLUID_ground_truth'] is False
    assert r['full_frame_recall'] is None