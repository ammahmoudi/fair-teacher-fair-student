from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import gin
import pandas as pd
import torch

from data_processing.ecg.dataset import (
    MitBihBeatDataset,
    build_stratified_record_level_split_assignments,
)
from distillation.core.ecg_classification_wrapper import ECGClassificationDistillationWrapper
from fairness.analyzers.ecg_classifier_fairness_analyzer import ECGClassifierFairnessAnalyzer
from models.ecg.time_llm_classifier import TimeLLMEcgClassifier
from scripts.time_llm.config_generator_mitbih import (
    generate_config_content as generate_classifier_config,
)
from scripts.time_llm.config_generator_mitbih_distillation import (
    generate_config_content,
    parse_length_configs,
)
from scripts.fairness.aggregate_mitbih_tuning_suite import (
    expected_variant_keys,
    seed_is_complete,
    selected_eo,
)
from scripts.fairness.aggregate_mitbih_multiseed_classification import (
    selected_eo as selected_multiseed_eo,
    validate_predictions,
)
from scripts.fairness.select_mitbih_binary_kd import CASE_PARAMETERS
from scripts.mitbih.prepare_beat_dataset import validate_existing_index_window_size


def _write_mini_ecg_index(directory: Path) -> tuple[Path, Path]:
    index_path = directory / "beat_index.csv"
    metadata_path = directory / "metadata_records.csv"
    rows = []
    for sample_index, class_id in zip((100, 460, 640, 1000, 1360), range(5)):
        rows.append(
            {
                "record_id": "100",
                "beat_sample_index": sample_index,
                "raw_symbol": "N",
                "aami_class": "NSVFQ"[class_id],
                "class_id": class_id,
                "sex": "F",
                "age_group": "50-69",
                "paced_group": "non_paced",
                "difficulty_group": "clean_or_mostly_clean",
                "split": "train",
                "signal_window": "0 1 0 -1",
            }
        )
    pd.DataFrame(rows).to_csv(index_path, index=False)
    pd.DataFrame([{"record_id": "100", "sex": "F"}]).to_csv(metadata_path, index=False)
    return index_path, metadata_path


def test_stratified_record_split_preserves_group_class_coverage():
    rows = []
    for record_index in range(12):
        sex = "F" if record_index % 2 == 0 else "M"
        for class_id in range(5):
            rows.extend(
                {
                    "record_id": str(100 + record_index),
                    "class_id": class_id,
                    "sex": sex,
                }
                for _ in range(class_id + 1)
            )
    index_rows = pd.DataFrame(rows)

    split_map = build_stratified_record_level_split_assignments(
        index_rows,
        seed=42,
        search_iterations=2_000,
    )

    assert Counter(split_map.values()) == {"train": 7, "val": 2, "test": 3}
    unique_rows = index_rows.drop_duplicates(["record_id", "class_id", "sex"])
    for split in ("train", "val", "test"):
        records = {record for record, assigned in split_map.items() if assigned == split}
        split_rows = unique_rows[unique_rows["record_id"].isin(records)]
        assert set(split_rows["sex"]) == {"F", "M"}
        for sex in ("F", "M"):
            assert set(split_rows.loc[split_rows["sex"] == sex, "class_id"]) == set(range(5))


def test_binary_label_modes_preserve_original_labels_and_rr_metadata():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        index_path, metadata_path = _write_mini_ecg_index(root)
        ectopy = MitBihBeatDataset(
            dataset_dir=root,
            split="train",
            window_size=4,
            beat_index_csv=index_path,
            metadata_csv=metadata_path,
            label_mode="binary_ectopy",
        )
        non_n = MitBihBeatDataset(
            dataset_dir=root,
            split="train",
            window_size=4,
            beat_index_csv=index_path,
            metadata_csv=metadata_path,
            label_mode="binary_non_n",
        )

    assert [sample.class_id for sample in ectopy.samples] == [0, 1, 1, 1]
    assert [sample.original_class_id for sample in ectopy.samples] == [0, 1, 2, 3]
    assert [sample.class_id for sample in non_n.samples] == [0, 1, 1, 1, 1]
    _, _, metadata = ectopy[1]
    assert metadata["label_mode"] == "binary_ectopy"
    assert metadata["original_class_id"] == 1
    assert abs(metadata["rr_prev_seconds"] - 1.0) < 1e-6
    assert abs(metadata["rr_next_seconds"] - 0.5) < 1e-6


def test_rr_feature_tensor_uses_log_seconds_and_neutral_fallback():
    metadata = {
        "rr_prev_seconds": torch.tensor([1.0, 2.0]),
        "rr_next_seconds": torch.tensor([0.5, 1.0]),
    }
    features = TimeLLMEcgClassifier._rr_feature_tensor(
        metadata,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    neutral = TimeLLMEcgClassifier._rr_feature_tensor(
        None,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(features, torch.log(torch.tensor([[1.0, 0.5], [2.0, 1.0]])))
    assert torch.equal(neutral, torch.zeros((2, 2)))

def test_cached_beat_index_must_match_requested_window_size():
    with TemporaryDirectory() as directory:
        index_path, _ = _write_mini_ecg_index(Path(directory))
        validate_existing_index_window_size(index_path, 4)
        try:
            validate_existing_index_window_size(index_path, 720)
        except ValueError as error:
            assert "has window size 4" in str(error)
        else:
            raise AssertionError("Expected a mismatched cached window size to be rejected")


def test_suite_selected_eo_uses_support_and_recall_qualification():
    predictions = pd.DataFrame(
        {
            "sex": ["F"] * 20 + ["M"] * 20,
            "y_true": [1] * 40,
            "y_pred": [1] * 20 + [1] * 15 + [0] * 5,
        }
    )
    assert abs(selected_eo(predictions, "sex", [1]) - 0.25) < 1e-6

    failed_predictions = predictions.copy()
    failed_predictions["y_pred"] = 0
    assert selected_eo(failed_predictions, "sex", [1]) is None
    assert selected_eo(predictions.iloc[:-1], "sex", [1]) is None

def test_suite_completion_requires_every_configured_model_variant():
    expected = expected_variant_keys({"distill_variants": ["baseline", "t1"]})

    assert expected == [
        "teacher",
        "student_baseline",
        "distilled_baseline",
        "distilled_t1",
    ]
    assert not seed_is_complete(["teacher"], expected)
    assert seed_is_complete(expected, expected)

def test_final_binary_aggregator_rejects_q_and_shared_failure():
    frame = pd.DataFrame(
        {
            "sex": ["F"] * 20 + ["M"] * 20,
            "y_true": [1] * 40,
            "y_pred": [0] * 40,
            "original_class_id": [1] * 40,
            "label_mode": ["binary_ectopy"] * 40,
        }
    )
    eo, details = selected_multiseed_eo(frame, "sex", [1], 20, 0.05)
    assert eo is None
    assert details["1"]["reportable"] is False

    frame.loc[0, "original_class_id"] = 4
    try:
        validate_predictions(frame, "binary_ectopy", Path("predictions.csv"))
    except ValueError as error:
        assert "contains Q beats" in str(error)
    else:
        raise AssertionError("Expected binary ectopy aggregation to reject Q beats")

def test_binary_kd_grid_contains_historical_ratio_and_temperature_candidates():
    assert CASE_PARAMETERS == {
        "binary_kd_a03_b03_t2": {"alpha": 0.3, "beta": 0.3, "temperature": 2.0},
        "binary_kd_a07_b03_t2": {"alpha": 0.7, "beta": 0.3, "temperature": 2.0},
        "binary_kd_a05_b05_t2": {"alpha": 0.5, "beta": 0.5, "temperature": 2.0},
        "binary_kd_a03_b07_t2": {"alpha": 0.3, "beta": 0.7, "temperature": 2.0},
        "binary_kd_a05_b05_t1": {"alpha": 0.5, "beta": 0.5, "temperature": 1.0},
        "binary_kd_a05_b05_t4": {"alpha": 0.5, "beta": 0.5, "temperature": 4.0},
    }
    assert CASE_PARAMETERS["binary_kd_a03_b03_t2"]["alpha"] == 0.3
    assert CASE_PARAMETERS["binary_kd_a03_b03_t2"]["beta"] == 0.3


def test_binary_rr_configs_set_two_classes_and_custom_context():
    length_set = parse_length_configs("720:32")[0]
    base_config = generate_classifier_config(
        mode="train",
        seed=42,
        llm_config={"llm_model": "BERT-tiny", "llm_dim": 128, "llm_layers": 2},
        length_set=length_set,
        label_mode="binary_ectopy",
        use_rr_features=True,
        beat_index_csv="./data/mit-bih-arrhythmia/beat_index_720.csv",
        class_balanced=True,
    )
    distill_config = generate_config_content(
        mode="train",
        seed=42,
        teacher_config={"llm_model": "BERT", "llm_dim": 768},
        student_config={"llm_model": "BERT-tiny", "llm_dim": 128, "llm_layers": 2},
        length_set=length_set,
        teacher_checkpoint_path="teacher.pth",
        label_mode="binary_ectopy",
        use_rr_features=True,
        teacher_use_rr_features=True,
        beat_index_csv="./data/mit-bih-arrhythmia/beat_index_720.csv",
    )

    for config in (base_config, distill_config):
        assert "'label_mode': 'binary_ectopy'" in config
        assert "'num_classes': 2" in config
        assert "'sequence_length': 720" in config
        assert "'patch_len': 32" in config
        assert "'use_rr_features': True" in config
        assert "beat_index_720.csv" in config
    assert "'teacher_use_rr_features': True" in distill_config

    with TemporaryDirectory() as directory:
        for name, config in (("base.gin", base_config), ("distill.gin", distill_config)):
            path = Path(directory) / name
            path.write_text(config.replace("LOGS_PLACEHOLDER", "logs"), encoding="utf-8")
            gin.clear_config()
            gin.parse_config_file(str(path), skip_unknown=True)
        gin.clear_config()


def test_joint_student_calibration_receives_gradients_and_round_trips():
    wrapper = ECGClassificationDistillationWrapper.__new__(ECGClassificationDistillationWrapper)
    wrapper.settings = {"num_classes": 5}
    wrapper.device = torch.device("cpu")
    wrapper.student_calibration_enabled = True
    wrapper.student_calibration_feature = "sex"
    wrapper.student_calibration_learning_rate = 1e-4
    wrapper.student_calibration_scale_regularization = 0.0
    wrapper.student_calibration_bias_regularization = 0.0
    wrapper.student_calibration_scale = None
    wrapper.student_calibration_bias = None
    wrapper._calibration_metadata = None
    wrapper._group_vocabs = {"sex": {"F": 0, "M": 1}}
    wrapper._initialize_student_calibration()

    logits = torch.randn(6, 5, requires_grad=True)
    metadata = {"sex": ["F", "M", "F", "M", "F", "M"]}
    calibrated = wrapper._apply_student_calibration(logits, metadata)
    torch.nn.functional.cross_entropy(
        calibrated,
        torch.tensor([0, 1, 2, 3, 4, 0]),
    ).backward()

    assert wrapper.student_calibration_scale.grad is not None
    assert wrapper.student_calibration_bias.grad is not None
    sidecar = wrapper._export_student_calibration_metadata()
    assert sidecar["training_mode"] == "joint"

    restored = ECGClassificationDistillationWrapper.__new__(ECGClassificationDistillationWrapper)
    restored.device = torch.device("cpu")
    restored.student_calibration_enabled = True
    restored.student_calibration_feature = "sex"
    restored.student_calibration_scale = None
    restored.student_calibration_bias = None
    restored._calibration_metadata = None
    restored._group_vocabs = {}
    restored._restore_student_calibration_metadata(sidecar)

    assert torch.allclose(
        calibrated.detach(),
        restored._apply_student_calibration(logits.detach(), metadata),
    )


def test_distillation_config_exposes_protocol_safe_tuning_controls():
    config = generate_config_content(
        mode="train_inference",
        seed=42,
        teacher_config={"llm_model": "BERT", "llm_dim": 768},
        student_config={"llm_model": "TinyBERT", "llm_dim": 312, "llm_layers": 4},
        length_set={
            "sequence_length": 256,
            "context_length": 256,
            "prediction_length": 0,
            "patch_len": 16,
        },
        teacher_checkpoint_path="teacher.pth",
        class_balanced=True,
        fair_teacher=True,
        fair_teacher_max_oversample=8.0,
        student_calibration=True,
        student_calibration_learning_rate=2.5e-5,
        student_calibration_scale_regularization=1e-3,
        student_calibration_bias_regularization=2e-3,
        checkpoint_selection="utility_fairness",
        checkpoint_selection_feature="sex",
        checkpoint_selection_min_s_recall=0.05,
        checkpoint_selection_require_s_recall=True,
    )

    assert "'class_balanced_sampling': True" in config
    assert "'fair_teacher_sampling': True" in config
    assert "'fair_teacher_max_oversample': 8.0" in config
    assert "'student_calibration_enabled': True" in config
    assert "'student_calibration_learning_rate': 2.5e-05" in config
    assert "'student_calibration_scale_regularization': 0.001" in config
    assert "'student_calibration_bias_regularization': 0.002" in config
    assert "'checkpoint_selection': 'utility_fairness'" in config
    assert "'checkpoint_selection_min_s_recall': 0.05" in config
    assert "'checkpoint_selection_require_s_recall': True" in config
    assert "'checkpoint_selection_min_best_group_recall': 0.05" in config
    assert "student_calibration_fairness_weight" not in config
    assert "student_calibration_max_abs_bias" not in config


def test_student_calibration_identity_regularization_receives_gradients():
    wrapper = ECGClassificationDistillationWrapper.__new__(ECGClassificationDistillationWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.student_calibration_enabled = True
    wrapper.student_calibration_scale_regularization = 0.5
    wrapper.student_calibration_bias_regularization = 0.25
    wrapper.student_calibration_scale = torch.nn.Parameter(torch.tensor([[2.0, 1.0]]))
    wrapper.student_calibration_bias = torch.nn.Parameter(torch.tensor([[1.0, 0.0]]))

    loss = wrapper._student_calibration_regularization_loss()
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.375))
    assert wrapper.student_calibration_scale.grad is not None
    assert wrapper.student_calibration_bias.grad is not None
    assert wrapper.student_calibration_scale.grad[0, 0] > 0
    assert wrapper.student_calibration_bias.grad[0, 0] > 0


def test_validation_checkpoint_selection_applies_utility_and_s_gates():
    wrapper = ECGClassificationDistillationWrapper.__new__(ECGClassificationDistillationWrapper)
    wrapper.checkpoint_selection_macro_f1_tolerance = 0.01
    wrapper.checkpoint_selection_min_s_recall = 0.05
    wrapper.checkpoint_selection_require_s_recall = True
    candidates = [
        {"epoch": 1, "macro_f1": 0.45, "s_recall": 0.04, "fairness_eo": 0.05, "val_loss": 0.8},
        {"epoch": 2, "macro_f1": 0.445, "s_recall": 0.08, "fairness_eo": 0.12, "val_loss": 0.7},
        {"epoch": 3, "macro_f1": 0.43, "s_recall": 0.10, "fairness_eo": 0.01, "val_loss": 0.6},
    ]

    selected = wrapper._select_validation_candidate(candidates)

    assert selected["epoch"] == 2


def test_validation_checkpoint_selection_can_require_s_recall():
    wrapper = ECGClassificationDistillationWrapper.__new__(ECGClassificationDistillationWrapper)
    wrapper.checkpoint_selection_macro_f1_tolerance = 0.01
    wrapper.checkpoint_selection_min_s_recall = 0.05
    wrapper.checkpoint_selection_require_s_recall = True
    candidates = [
        {"epoch": 1, "macro_f1": 0.45, "s_recall": 0.01, "fairness_eo": 0.05, "val_loss": 0.8},
        {"epoch": 2, "macro_f1": 0.44, "s_recall": 0.02, "fairness_eo": 0.04, "val_loss": 0.7},
    ]

    try:
        wrapper._select_validation_candidate(candidates)
    except RuntimeError as error:
        assert "S-recall gate" in str(error)
    else:
        raise AssertionError("Expected the required S-recall gate to reject all candidates")

def test_validation_fairness_excludes_shared_class_failure():
    class IdentityWithMetadata(torch.nn.Module):
        def forward(self, inputs, metadata=None):
            return inputs

    wrapper = ECGClassificationDistillationWrapper.__new__(ECGClassificationDistillationWrapper)
    wrapper.student = IdentityWithMetadata()
    wrapper.settings = {"num_classes": 2}
    wrapper.device = torch.device("cpu")
    wrapper.checkpoint_selection_feature = "sex"
    wrapper.checkpoint_selection_s_class = 1
    wrapper.checkpoint_selection_fairness_classes = [1]
    wrapper.checkpoint_selection_min_group_class_support = 2
    wrapper.checkpoint_selection_min_best_group_recall = 0.05
    wrapper._apply_student_calibration = lambda logits, metadata: logits
    loader = [
        (
            torch.tensor([[5.0, 0.0]] * 4),
            torch.tensor([1, 1, 1, 1]),
            {"sex": ["F", "F", "M", "M"]},
        )
    ]

    metrics = wrapper._validation_selection_metrics(loader)

    assert metrics["s_recall"] == 0.0
    assert metrics["class_eo"] == {}
    assert metrics["fairness_eo"] is None


def test_fairness_report_excludes_low_support_eo_cells():
    rows = []
    for sex, class_id, support in (("F", 0, 25), ("M", 0, 25), ("F", 1, 2), ("M", 1, 25)):
        rows.extend(
            {"sex": sex, "y_true": class_id, "y_pred": class_id}
            for _ in range(support)
        )
    with TemporaryDirectory() as directory:
        prediction_path = Path(directory) / "predictions.csv"
        pd.DataFrame(rows).to_csv(prediction_path, index=False)
        report = ECGClassifierFairnessAnalyzer(
            prediction_path,
            group_column="sex",
            min_group_class_support=20,
        ).analyze()

    assert report["classwise_one_vs_rest"]["0"]["eo_reportable"] is True
    assert report["classwise_one_vs_rest"]["1"]["eo_reportable"] is False
    assert report["classwise_one_vs_rest"]["1"]["eo_gap"] is None
    assert report["summary"]["reportable_eo_classes"] == [0]
    assert report["summary"]["excluded_eo_classes"] == [1]


def test_fairness_report_excludes_classes_with_no_effective_recall():
    rows = []
    for sex in ("F", "M"):
        rows.extend({"sex": sex, "y_true": 0, "y_pred": 0} for _ in range(25))
        rows.extend({"sex": sex, "y_true": 1, "y_pred": 0} for _ in range(25))
    with TemporaryDirectory() as directory:
        prediction_path = Path(directory) / "predictions.csv"
        pd.DataFrame(rows).to_csv(prediction_path, index=False)
        report = ECGClassifierFairnessAnalyzer(prediction_path, group_column="sex").analyze()

    class_report = report["classwise_one_vs_rest"]["1"]
    assert class_report["group_support"] == {"F": 25, "M": 25}
    assert class_report["eo_reportable"] is False
    assert class_report["eo_exclusion_reasons"] == ["insufficient_class_recall"]
    assert class_report["raw_eo_gap"] == 0.0
    assert class_report["eo_gap"] is None
