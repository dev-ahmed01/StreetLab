"""One-command W04: atomic output, real human provenance, incomplete model fail-closed."""
from pathlib import Path

import pytest

from streetlab_phase3.video.w04_integrated_review_run import run_w04_reviewed_box_batch

BUNDLE=Path('/mnt/data/W04_COMPLETE_CONTAINER_EXPERIMENT_INPUT_01.zip')
REVIEW=Path('/mnt/data/W04_BLIND_REVIEW_15_02')
CONSENSUS=Path('/mnt/data/W04_manual_dual_consensus_20261009_01.json')

@pytest.fixture
def user_w04():
    if not BUNDLE.exists() or not REVIEW.exists() or not CONSENSUS.exists():
        pytest.skip('Private user-provided W04 data unavailable; not fetched by CI')
    return dict(bundle=BUNDLE,review_dir=REVIEW,consensus_file=CONSENSUS)


def test_cached_pipeline_real_w04_and_no_overwrite(user_w04,tmp_path):
    target=tmp_path/'cached'
    report=run_w04_reviewed_box_batch(**user_w04,output_dir=target,cached_only=True)
    assert report['status']=='W04_SINGLE_BATCH_CACHED_ONLY'
    assert report['reviewed_cases']==13
    assert report['unresolved_cases']==2
    assert report['cached_review_summary']['openvino']['localized_compatible_class_cases']==12
    assert sorted(x.name for x in target.iterdir())==['cached_review.json','one_batch_summary.json']
    with pytest.raises(FileExistsError):
        run_w04_reviewed_box_batch(**user_w04,output_dir=target,cached_only=True)


def test_real_inference_never_runs_without_complete_original_export(user_w04,tmp_path):
    target=tmp_path/'real'
    with pytest.raises(ValueError,match='requires --model-dir'):
        run_w04_reviewed_box_batch(**user_w04,output_dir=target)
    assert not target.exists()
    folder=tmp_path/'incomplete_model'
    folder.mkdir()
    (folder/'model.xml').write_text('fake')
    with pytest.raises(ValueError,match='differs from SHA-verified W04 export'):
        run_w04_reviewed_box_batch(**user_w04,output_dir=target,model_dir=folder)
    assert not target.exists()


def test_refuse_wrong_or_missing_video(user_w04,tmp_path):
    with pytest.raises(FileNotFoundError,match='Actual source video path'):
        run_w04_reviewed_box_batch(**user_w04,output_dir=tmp_path/'missing',
                                   video=tmp_path/'not_found.avi',cached_only=True)