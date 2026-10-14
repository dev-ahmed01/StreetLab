# W04 reviewer accepted cases × cached OpenVINO/PyTorch: measured replay

**Status: verified case-level development evidence; NOT physically adjudicated population precision, not unseen validation, no production promotion.**

On 9 Oct 2026, the user's SHA-verified W04 source archive and the independent two-reviewer Build 2 consensus were joined using the **centers actually saved** in the original 21-frame OpenVINO and PyTorch SAHI detector audits. Both source detector CSVs and report JSONs were checked against the bundle's SHA-256 file manifest, and the human review package/consensus were checked against their original manifest SHA. 13 of 15 submitted cases reached the frozen two-reviewer PRESENT / class / box-IoU consensus; 2 remain unresolved.

Each case was counted as *localized* when **any cached detector center was inside the two reviewers' agreed physical box**, and as *class-compatible* when at least one such center had the same class as the independent visual consensus. No FLUID labels were changed or used to determine any candidate selection. No model inference or tracker replay occurred.

| W04 accepted physical-case test | OpenVINO | PyTorch |
|---|---:|---:|
| Agreed-present reviewed cases | 13 | 13 |
| >=1 detector center inside reviewed box | **13** | **12** |
| >=1 detector center with reviewed physical class | **12** | **11** |
| >1 detector center inside reviewed box | **5** | **5** |
| Reviewed class not present in 4-class model ontology | 1 AUTO_RICKSHAW | 1 AUTO_RICKSHAW |

Frame 10990 case 1 (MOTORCYCLE) is inside the agreed physical box for OpenVINO but outside for PyTorch. Five cases have multiple same-class model centers inside independently reviewed boxes: W04 cases 4,5,6 (motorcycles), 10,11 (cars). These are **potential duplicate evidence**, not proof from centers alone that the model boxes overlap or reference one physical object. In case 9, both reviewers identified AUTO_RICKSHAW; both models predicted CAR and nearest FLUID label was `moped`: a real ontology limitation, not a justification for blindly remapping all cars.

The original four-class FLUID comparison remains frozen: **OpenVINO 670/857 precision = 78.18%; 670/739 recall = 90.66%**. The selected 13 reviewed cases cannot estimate overall physical precision/recall, and two unresolved cases are not counted as agreement.

## Next actual evidence run

Run the integrated **25-policy real box** experiment on the actual 21 W04 frames with the verified ORIGINAL OpenVINO export directory and actual local review artifacts. A verified ZIP contains the model .xml and .bin, but not all original model-tree metadata: the reconstructed model-tree SHA differs. **Do not** treat just the archived XML/BIN as the full original model. Original path from W04 report: `artifacts/phase3/sahi_detector_trials/W04_openvino_export640_01/checkpoint_openvino_model`.

For reproducible code and exact one-command Windows invocation see the separate source/runbook bundle from this phase of work. The real capture must generate per-tile xyxy boxes and compare Hard NMS, Soft-NMS, weighted fusion and unmerged control against accepted physical boxes; output remains development-only. The 21 sampled frames are 30 frames apart and CANNOT measure ByteTrack continuity or ID switches. Build 3 continuous 201-frame tracking and Build 4 untouched holdout with real same-window T000 baseline are still required.

**PR #10 stays draft**. No production Geo-trax/T000/P3B/original FLUID modification, no invented reviewer decisions.
