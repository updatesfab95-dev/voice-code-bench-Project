from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def summarize_examples(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize dataset rows into counts for duration, domain, difficulty, and entity types."""
    rows = list(rows)

    total_duration = 0.0
    domain_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    entity_type_counts: Counter[str] = Counter()
    total_entities = 0
    entity_count_overall = 0

    for row in rows:
        duration = row.get("duration")
        if isinstance(duration, (int, float)):
            total_duration += float(duration)

        domain = row.get("domain")
        if domain:
            domain_counts[str(domain)] += 1

        difficulty = row.get("difficulty")
        if difficulty:
            difficulty_counts[str(difficulty)] += 1

        entities = row.get("entities") or []
        entity_count_overall += len(entities)
        total_entities += len(entities)
        for item in entities:
            if isinstance(item, Mapping):
                entity_type = item.get("type")
                if entity_type:
                    entity_type_counts[str(entity_type)] += 1

    average_entity_count = round((entity_count_overall / len(rows)) if rows else 0.0, 2)

    return {
        "num_examples": len(rows),
        "total_duration_seconds": round(total_duration, 3),
        "total_entities": total_entities,
        "average_entity_count": average_entity_count,
        "domains": dict(sorted(domain_counts.items())),
        "difficulty": dict(sorted(difficulty_counts.items())),
        "entity_types": dict(sorted(entity_type_counts.items())),
    }


def preview_examples(rows: Iterable[Mapping[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Create a compact preview of the first few dataset rows."""
    preview = []
    for row in list(rows)[:limit]:
        entities = row.get("entities") or []
        transcript = (row.get("transcripts") or {})
        preview.append(
            {
                "file_name": row.get("file_name"),
                "domain": row.get("domain"),
                "difficulty": row.get("difficulty"),
                "entity_count": len(entities),
                "template": transcript.get("template"),
                "entity_types": sorted(
                    {
                        item["type"]
                        for item in entities
                        if isinstance(item, Mapping) and item.get("type")
                    }
                ),
            }
        )
    return preview


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_matches_prediction(entity: Mapping[str, Any], prediction: str) -> bool:
    """Return True if any canonical or acoustic representation appears in the prediction text."""
    if not isinstance(entity, Mapping):
        return False

    normalized_prediction = _normalize_text(prediction)
    if not normalized_prediction:
        return False

    references = []
    for key in ("canonical", "acoustic", "value"):
        value = entity.get(key)
        if value is not None:
            references.append(str(value))

    for ref in references:
        if _normalize_text(ref) in normalized_prediction:
            return True
    return False


def score_predictions(rows: Iterable[Mapping[str, Any]], predictions: Mapping[str, str] | Sequence[str] | None = None) -> dict[str, Any]:
    """Compute CTEM and TSR-style metrics from model predictions."""
    rows = list(rows)
    if predictions is None:
        predictions = ["" for _ in rows]

    if isinstance(predictions, Mapping):
        prediction_lookup = dict(predictions)
    else:
        prediction_lookup = {}
        for index, row in enumerate(rows):
            file_name = row.get("file_name") or f"row_{index}"
            if index < len(predictions):
                prediction_lookup[file_name] = str(predictions[index])

    total_correct_entities = 0
    total_entities = 0
    success_count = 0

    for index, row in enumerate(rows):
        file_name = row.get("file_name") or f"row_{index}"
        prediction = prediction_lookup.get(file_name, "")
        entities = row.get("entities") or []
        if not entities:
            continue

        total_entities += len(entities)
        record_correct = 0
        for entity in entities:
            if entity_matches_prediction(entity, prediction):
                total_correct_entities += 1
                record_correct += 1

        if record_correct == len(entities):
            success_count += 1

    num_examples = len(rows)
    ctem = (total_correct_entities / total_entities) if total_entities else 0.0
    tsr = (success_count / num_examples) if num_examples else 0.0

    return {
        "num_examples": num_examples,
        "num_entities": total_entities,
        "correct_entities": total_correct_entities,
        "ctem": round(ctem, 4),
        "tsr": round(tsr, 4),
        "success_examples": success_count,
    }


def export_summary_csv(rows: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write a compact summary CSV for the dataset rows."""
    rows = list(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file_name", "duration", "domain", "difficulty", "entity_count", "entity_types"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            entities = row.get("entities") or []
            entity_types = sorted(
                {
                    item["type"]
                    for item in entities
                    if isinstance(item, Mapping) and item.get("type")
                }
            )
            writer.writerow(
                {
                    "file_name": row.get("file_name", ""),
                    "duration": row.get("duration", ""),
                    "domain": row.get("domain", ""),
                    "difficulty": row.get("difficulty", ""),
                    "entity_count": len(entities),
                    "entity_types": "; ".join(entity_types),
                }
            )

    return path


def load_voice_code_bench(split: str = "test", limit: int | None = None) -> list[dict[str, Any]]:
    """Load the VoiceCodeBench dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("Install project dependencies with: python -m pip install -e .") from exc

    ds = load_dataset("besimple-ai/voice-code-bench", split=split)
    if limit is None:
        rows = ds
    else:
        rows = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VoiceCodeBench samples.")
    parser.add_argument("--split", default="test", help="Dataset split to load (default: test)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to inspect")
    parser.add_argument("--preview", type=int, default=0, help="Optional number of example rows to preview")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    parser.add_argument("--export-csv", type=str, default=None, help="Optional CSV path to write a compact row summary")
    parser.add_argument("--prediction-file", type=str, default=None, help="Optional JSON file mapping file_name -> prediction text")
    args = parser.parse_args()

    rows = load_voice_code_bench(split=args.split, limit=args.limit)
    summary = summarize_examples(rows)

    if args.prediction_file:
        with open(args.prediction_file, "r", encoding="utf-8") as fh:
            predictions = json.load(fh)
        metrics = score_predictions(rows, predictions)
        summary["scoring"] = metrics

    if args.export_csv:
        export_summary_csv(rows, args.export_csv)
        summary["csv_export"] = args.export_csv

    if args.json:
        payload = {**summary}
        if args.preview > 0:
            payload["preview"] = preview_examples(rows, limit=args.preview)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"Examples: {summary['num_examples']}")
    print(f"Total duration (s): {summary['total_duration_seconds']}")
    print(f"Total entities: {summary['total_entities']}")
    print(f"Average entities per example: {summary['average_entity_count']}")
    print("Domains:")
    for key, value in summary["domains"].items():
        print(f"  - {key}: {value}")
    print("Difficulty:")
    for key, value in summary["difficulty"].items():
        print(f"  - {key}: {value}")
    print("Entity types:")
    for key, value in summary["entity_types"].items():
        print(f"  - {key}: {value}")

    if args.prediction_file:
        print("Scoring:")
        metrics = summary["scoring"]
        print(f"  - correct entities: {metrics['correct_entities']} / {metrics['num_entities']}")
        print(f"  - CTEM: {metrics['ctem']}")
        print(f"  - TSR: {metrics['tsr']}")

    if args.export_csv:
        print(f"CSV export written to: {args.export_csv}")

    if args.preview > 0:
        print("Preview:")
        for item in preview_examples(rows, limit=args.preview):
            print(f"  - file={item['file_name']} domain={item['domain']} difficulty={item['difficulty']} entity_count={item['entity_count']} types={item['entity_types']}")


if __name__ == "__main__":
    main()
