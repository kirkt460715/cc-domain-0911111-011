import json
import tempfile
from pathlib import Path
from .service import Service
from .ingress.readers import read_jsonl
from .output.snapshots import verify_export


def demonstration():
    with tempfile.TemporaryDirectory(prefix="liangheng-") as directory:
        root = Path(directory)
        with Service(root / "lab.sqlite3") as app:
            app.configuration.apply(
                {
                    "id": "demo-config",
                    "instruments": [
                        {
                            "id": "press-1",
                            "label": "Synthetic pressure instrument",
                            "channels": {"p": "kPa"},
                        },
                        {
                            "id": "press-2",
                            "label": "Synthetic pressure instrument 2",
                            "channels": {"p": "kPa"},
                        },
                    ],
                    "sources": [{"id": "bench", "allowed_lateness": 0}],
                    "certificates": [
                        {
                            "id": "cert-1",
                            "instrument": "press-1",
                            "channel": "p",
                            "valid_from": "2026-01-01T00:00:00Z",
                            "valid_to": "2027-01-01T00:00:00Z",
                            "input_unit": "kPa",
                            "output_unit": "kPa",
                            "slope": "1",
                            "intercept": ".1",
                            "uncertainty": ".01",
                            "minimum": "0",
                            "maximum": "100",
                            "issued_at": "2025-12-01T00:00:00Z",
                        },
                        {
                            "id": "cert-2",
                            "instrument": "press-2",
                            "channel": "p",
                            "valid_from": "2026-01-01T00:00:00Z",
                            "valid_to": "2027-01-01T00:00:00Z",
                            "input_unit": "kPa",
                            "output_unit": "kPa",
                            "slope": "1",
                            "intercept": "0",
                            "uncertainty": ".02",
                            "minimum": "0",
                            "maximum": "100",
                            "issued_at": "2025-12-01T00:00:00Z",
                        },
                    ],
                    "policies": [
                        {
                            "id": "q1",
                            "unit": "kPa",
                            "lower": "0",
                            "upper": "20",
                            "required_tags": ["lot"],
                        },
                        {"id": "q2", "unit": "kPa", "lower": "0", "upper": "20"},
                    ],
                    "bindings": [
                        {"instrument": "press-1", "channel": "p", "policy": "q1"},
                        {"instrument": "press-2", "channel": "p", "policy": "q2"},
                    ],
                }
            )
            values = [
                {
                    "id": f"r{i}",
                    "source": "bench",
                    "sequence": i,
                    "instrument": "press-1",
                    "channel": "p",
                    "event_time": f"2026-11-11T00:00:{i * 10:02d}Z",
                    "value": str(10 + i),
                    "unit": "kPa",
                    "uncertainty": "0",
                    "tags": {"lot": "synthetic-A"},
                }
                for i in range(3)
            ]
            # The second channel shares the same reference standard, so its
            # systematic uncertainty is correlated with the first channel.
            second = [
                {
                    "id": f"s{i}",
                    "source": "bench",
                    "sequence": 100 + i,
                    "instrument": "press-2",
                    "channel": "p",
                    "event_time": f"2026-11-11T00:00:{i * 10:02d}Z",
                    "value": str(5 + i),
                    "unit": "kPa",
                    "uncertainty": "0",
                    "tags": {"lot": "synthetic-A"},
                }
                for i in range(3)
            ]
            rows = read_jsonl("\n".join(json.dumps(value) for value in values))
            ingestion = app.pipeline.ingest(
                "demo-batch", rows, received_at="2026-11-11T00:01:00Z"
            )
            duplicate = app.pipeline.ingest(
                "demo-batch", rows, received_at="2026-11-11T00:01:00Z"
            )
            second_ingestion = app.pipeline.ingest(
                "demo-batch-2",
                read_jsonl("\n".join(json.dumps(value) for value in second)),
                received_at="2026-11-11T00:01:00Z",
            )
            app.catalog.watermark("bench", "2026-11-11T00:01:00Z")
            window = app.windows.materialize(
                {
                    "id": "minute",
                    "instrument": "press-1",
                    "channel": "p",
                    "start": "2026-11-11T00:00:00Z",
                    "end": "2026-11-11T00:01:00Z",
                    "seconds": 60,
                    "sources": ["bench"],
                }
            )
            app.windows.seal("minute", window["generation"])
            snapshot = app.snapshots.create("release-1", ["minute"])
            app.snapshots.export("release-1", root / "snapshot.json")
            verification = verify_export(root / "snapshot.json")
            integrity = app.inspector.check()
            assert ingestion["accepted"] == 3
            assert duplicate["reused"] is True
            assert second_ingestion["accepted"] == 3
            assert window["windows"][0]["statistics"]["mean"] == "11.1"
            assert verification["observations"] == 3
            model_result = _measurement_model(app)
            return {
                "ingestion": ingestion,
                "duplicate_reused": duplicate["reused"],
                "mean": window["windows"][0]["statistics"]["mean"],
                "snapshot": snapshot,
                "export_verified": verification,
                "integrity": integrity,
                "measurement_model": model_result,
            }


def _measurement_model(app):
    registered = app.models.register_model(
        {
            "id": "composite-pressure",
            "description": "sum of two pressure channels sharing a reference",
            "expression": "left + right",
            "output_unit": "kPa",
            "inputs": [
                {
                    "name": "left",
                    "unit": "kPa",
                    "observation": "r0",
                    "certificate": "cert-1",
                },
                {
                    "name": "right",
                    "unit": "kPa",
                    "observation": "s0",
                    "certificate": "cert-2",
                },
            ],
        },
        created_at="2026-11-11T01:00:00Z",
    )
    report = app.models.evaluate(
        "composite-pressure",
        {
            "coverage_factor": "2",
            "effects": [
                {
                    "id": "shared-reference",
                    "uncertainty": "0.05",
                    "unit": "kPa",
                    "certificate": "cert-1",
                    "label": "common reference standard",
                }
            ],
            "inputs": {
                "left": {
                    "random_uncertainty": "0.03",
                    "effects": [{"effect": "shared-reference"}],
                },
                "right": {
                    "random_uncertainty": "0.04",
                    "effects": [{"effect": "shared-reference"}],
                },
            },
            "monte_carlo": {"seed": 20261111, "samples": 20000},
        },
        report_id="model-report-1",
        created_at="2026-11-11T02:00:00Z",
    )
    return {
        "model_version": registered["version"],
        "combined_value": report["combined_value"],
        "standard_uncertainty": report["standard_uncertainty"],
        "expanded_uncertainty": report["expanded_uncertainty"],
        "coverage_factor": report["coverage_factor"],
        "coverage_note": report["coverage_note"],
        "systematic_variance": report["contributions"]["systematic"]["variance"],
        "monte_carlo_valid_samples": report["monte_carlo"]["valid_samples"],
        "monte_carlo_invalid_samples": report["monte_carlo"]["invalid_samples"],
        "sampling_comparison": report["monte_carlo"]["comparison"],
    }
