# Test Coverage Forensic Audit

This document classifies all active tests in the SIH26158 repository, distinguishes verified contracts from uninstantiated engines, and documents intentional testing boundaries for Phase 0.

---

## 1. Test Classification Matrix

| Test Function | Test File | Classification | Verification Scope | Status |
| :--- | :--- | :--- | :--- | :--- |
| `test_root_package_import` | `tests/unit/test_imports.py` | **UNIT** | Version string and root package namespace | PASSED |
| `test_all_submodules_importable` | `tests/unit/test_imports.py` | **UNIT** | Importability of all 10 core modules | PASSED |
| `test_telemetry_record_immutability` | `tests/unit/test_data_contracts.py` | **UNIT** | Immutability/frozen enforcement of telemetry data | PASSED |
| `test_camera_intrinsics_matrix` | `tests/unit/test_data_contracts.py` | **UNIT** | $3 \times 3$ intrinsic projection matrix calculation | PASSED |
| `test_frame_quality_score` | `tests/unit/test_data_contracts.py` | **UNIT** | Quality metrics instantiation and bounds | PASSED |
| `test_sim3_transform_contract` | `tests/unit/test_data_contracts.py` | **UNIT** | 7-DoF Sim(3) parameter contract integrity | PASSED |
| `test_uncertainty_field_contract` | `tests/unit/test_data_contracts.py` | **UNIT** | Per-vertex uncertainty and unobserved flags | PASSED |
| `test_valid_accuracy_metric_creation` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rule 3 compliance on valid provenance record | PASSED |
| `test_reject_empty_metric_name` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of empty metric name | PASSED |
| `test_reject_empty_dataset_name` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of claims without named dataset | PASSED |
| `test_reject_empty_ground_truth_reference` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of claims without ground truth ref | PASSED |
| `test_reject_empty_calculation_method` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of claims without calculation script | PASSED |
| `test_reject_empty_unit` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of claims without unit | PASSED |
| `test_reject_none_provenance` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of claims with missing provenance object | PASSED |
| `test_reject_nan_metric_value` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of NaN numeric results | PASSED |
| `test_reject_infinite_metric_value` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of Infinite ($\pm \infty$) numeric results | PASSED |
| `test_reject_negative_error_metric` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of negative distances/errors (RMSE, Chamfer) | PASSED |
| `test_reject_out_of_bounds_percentage` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of percentage values $<0\%$ or $>100\%$ | PASSED |
| `test_reject_invalid_threshold_tau` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Rejection of non-positive threshold $\tau$ | PASSED |
| `test_strict_metric_validator_helper` | `tests/unit/test_validation_rules.py` | **SCIENTIFIC VALIDATION** | Convenience validator integrity and checksum checks | PASSED |
| `test_pipeline_configuration_compatibility`| `tests/integration/test_pipeline_interface.py`| **INTERFACE** | Top-level config translation to submodules | PASSED |
| `test_pipeline_status_transitions` | `tests/integration/test_pipeline_interface.py`| **INTEGRATION** | Pipeline stage state machine transitions | PASSED |

---

## 2. Intentionally Deferred Testing (Phase 1 & Phase 2)

The following components are **intentionally uninstantiated and not tested** in Phase 0:
1. **Live Video Demuxing & Codec Decoding:** Deferred to Phase 1 (Ingestion).
2. **Dense Pointmap AI Inference:** Deferred to Phase 2 (Foundation AI Geometry).
3. **Multi-View Global Mesh Reconstruction:** Deferred to Phase 3 (Surface Meshing).
4. **End-to-End LiDAR Point Cloud Regression:** Requires physical test dataset acquisition in Phase 1.
