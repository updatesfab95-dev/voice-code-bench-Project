from voice_code_bench.dataset import export_summary_csv, score_prediction_directory, score_predictions, summarize_examples


def test_summarize_examples_counts_basic_metrics():
    rows = [
        {
            "duration": 5.5,
            "domain": "devops",
            "difficulty": "easy",
            "entities": [
                {"type": "flag"},
                {"type": "file_path"},
                {"type": "flag"},
            ],
        },
        {
            "duration": 2.5,
            "domain": "support",
            "difficulty": "medium",
            "entities": [
                {"type": "email"},
                {"type": "url"},
            ],
        },
    ]

    summary = summarize_examples(rows)

    assert summary["num_examples"] == 2
    assert summary["total_duration_seconds"] == 8.0
    assert summary["total_entities"] == 5
    assert summary["domains"]["devops"] == 1
    assert summary["domains"]["support"] == 1
    assert summary["entity_types"]["email"] == 1
    assert summary["entity_types"]["flag"] == 2
    assert summary["entity_types"]["url"] == 1
    assert summary["difficulty"]["easy"] == 1
    assert summary["difficulty"]["medium"] == 1
    assert summary["average_entity_count"] == 2.5


def test_score_predictions_reports_ctem_and_tsr():
    rows = [
        {
            "file_name": "a.wav",
            "entities": [
                {"canonical": "212-555-0100", "acoustic": "two one two five five five zero one zero zero"},
                {"canonical": "https://example.com", "acoustic": "https example dot com"},
            ],
        },
        {
            "file_name": "b.wav",
            "entities": [
                {"canonical": "--dry-run", "acoustic": "double dash dry dash run"},
            ],
        },
    ]

    metrics = score_predictions(rows, {
        "a.wav": "call me at two one two five five five zero one zero zero and visit https example dot com",
        "b.wav": "double dash dry dash run",
    })

    assert metrics["num_examples"] == 2
    assert metrics["num_entities"] == 3
    assert metrics["correct_entities"] == 3
    assert metrics["ctem"] == 1.0
    assert metrics["tsr"] == 1.0


def test_export_summary_csv(tmp_path):
    rows = [
        {
            "file_name": "sample.wav",
            "duration": 12.5,
            "domain": "support",
            "difficulty": "medium",
            "entities": [{"type": "phone_number", "canonical": "212-555-0100"}],
        }
    ]

    output = tmp_path / "summary.csv"
    export_summary_csv(rows, output)

    assert output.exists()
    content = output.read_text()
    assert "file_name" in content
    assert "sample.wav" in content
    assert "support" in content


def test_score_prediction_directory(tmp_path):
    rows = [
        {
            "file_name": "a.wav",
            "entities": [{"canonical": "--dry-run", "acoustic": "double dash dry dash run"}],
        },
        {
            "file_name": "b.wav",
            "entities": [{"canonical": "https://example.com", "acoustic": "https example dot com"}],
        },
    ]

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    (pred_dir / "model_one.json").write_text(
        '{"a.wav": "double dash dry dash run", "b.wav": "https example dot com"}',
        encoding="utf-8",
    )
    (pred_dir / "model_two.json").write_text(
        '{"a.wav": "double dash dry run", "b.wav": "https example google com"}',
        encoding="utf-8",
    )

    results = score_prediction_directory(rows, pred_dir)

    assert len(results) == 2
    assert results[0]["model_name"] == "model_one"
    assert results[0]["ctem"] == 1.0
    assert results[1]["model_name"] == "model_two"
    assert results[1]["ctem"] < 1.0
