"""Persistence for versioned measurement models and immutable reports."""

import datetime as dt

from ..errors import Conflict, Missing
from ..ledger.audit import append
from ..values import (
    canonical,
    identifier,
    instant,
    strict_json,
    time_text,
)
from .models import MeasurementModel

MODEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurement_models(
    id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    created TEXT NOT NULL,
    PRIMARY KEY(id, version)
);
CREATE TABLE IF NOT EXISTS model_reports(
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    supersedes TEXT,
    created TEXT NOT NULL,
    FOREIGN KEY(model_id, model_version) REFERENCES measurement_models(id, version)
);
CREATE INDEX IF NOT EXISTS model_report_model ON model_reports(model_id, model_version);
"""


class ModelStore:
    def __init__(self, database, catalog):
        self.db = database
        self.catalog = catalog
        # CREATE TABLE IF NOT EXISTS is idempotent; executescript commits any
        # open transaction, so it must run outside the database transaction.
        database.connection.executescript(MODEL_SCHEMA)

    def register_model(self, data, created_at):
        registry = self.catalog.registry()
        model_id = identifier(data["id"]) if isinstance(data, dict) else data
        with self.db.transaction() as con:
            rows = con.execute(
                "SELECT version,payload FROM measurement_models WHERE id=? ORDER BY version",
                (model_id,),
            ).fetchall()
            all_rows = con.execute(
                "SELECT payload FROM measurement_models"
            ).fetchall()
            existing = {}
            for row in all_rows:
                model = MeasurementModel.restore(row["payload"], registry)
                existing[model.id] = model
            next_version = (
                max((row["version"] for row in rows), default=0)
            ) + 1
            model = MeasurementModel.parse(
                data,
                registry,
                version=next_version,
                existing=existing,
            )
            for row in rows:
                previous = MeasurementModel.restore(row["payload"], registry)
                if previous.fingerprint == model.fingerprint:
                    return {
                        "id": previous.id,
                        "version": previous.version,
                        "created": False,
                        "fingerprint": previous.fingerprint,
                    }
            timestamp = time_text(instant(created_at))
            con.execute(
                "INSERT INTO measurement_models VALUES(?,?,?,?,?)",
                (
                    model.id,
                    model.version,
                    canonical(model.as_dict()),
                    model.fingerprint,
                    timestamp,
                ),
            )
            append(
                con,
                "measurement_model_registered",
                model.id,
                {"version": model.version, "fingerprint": model.fingerprint},
            )
            return {
                "id": model.id,
                "version": model.version,
                "created": True,
                "fingerprint": model.fingerprint,
            }

    def get_model(self, model_id, version=None):
        registry = self.catalog.registry()
        if version is None:
            row = self.db.one(
                "SELECT payload FROM measurement_models WHERE id=? ORDER BY version DESC LIMIT 1",
                (model_id,),
            )
        else:
            row = self.db.one(
                "SELECT payload FROM measurement_models WHERE id=? AND version=?",
                (model_id, version),
            )
        if row is None:
            raise Missing("measurement model not registered", field=model_id)
        return MeasurementModel.restore(row["payload"], registry)

    def list_models(self):
        rows = self.db.rows(
            "SELECT id,version,fingerprint,created FROM measurement_models ORDER BY id,version"
        )
        return rows

    def save_report(self, report, *, created_at):
        timestamp = (
            time_text(created_at)
            if isinstance(created_at, dt.datetime)
            else time_text(instant(created_at))
        )
        with self.db.transaction() as con:
            if con.execute(
                "SELECT id FROM model_reports WHERE id=?", (report["id"],)
            ).fetchone():
                raise Conflict("model report id already exists")
            if report.get("supersedes"):
                old = con.execute(
                    "SELECT payload FROM model_reports WHERE id=?",
                    (report["supersedes"],),
                ).fetchone()
                if old is None:
                    raise Missing(
                        "superseded report not found", field=report["supersedes"]
                    )
            con.execute(
                "INSERT INTO model_reports VALUES(?,?,?,?,?,?,?)",
                (
                    report["id"],
                    report["model_id"],
                    report["model_version"],
                    canonical(report),
                    report["fingerprint"],
                    report["supersedes"],
                    timestamp,
                ),
            )
            append(
                con,
                "model_report_issued",
                report["id"],
                {
                    "model_id": report["model_id"],
                    "model_version": report["model_version"],
                    "supersedes": report["supersedes"],
                    "fingerprint": report["fingerprint"],
                },
            )
        return {"id": report["id"], "created": True, "fingerprint": report["fingerprint"]}

    def get_report(self, report_id):
        row = self.db.one(
            "SELECT payload,fingerprint FROM model_reports WHERE id=?",
            (report_id,),
        )
        if row is None:
            raise Missing("model report not found", field=report_id)
        payload = strict_json(row["payload"])
        if payload["fingerprint"] != row["fingerprint"]:
            from ..errors import IntegrityError

            raise IntegrityError(
                "stored model report fingerprint mismatch",
                details={"report": report_id},
            )
        return payload

    def list_reports(self, model_id=None):
        if model_id is None:
            rows = self.db.rows(
                "SELECT id,model_id,model_version,supersedes,created,fingerprint FROM model_reports ORDER BY created,id"
            )
        else:
            rows = self.db.rows(
                "SELECT id,model_id,model_version,supersedes,created,fingerprint FROM model_reports WHERE model_id=? ORDER BY created,id",
                (model_id,),
            )
        return rows

    def reports_for_certificate(self, certificate_id):
        rows = self.db.rows(
            "SELECT id,model_id,model_version,payload,supersedes FROM model_reports ORDER BY created,id"
        )
        references = []
        for row in rows:
            payload = strict_json(row["payload"])
            certificates = {
                item["certificate"] for item in payload.get("inputs", [])
            }
            certificates |= {
                effect["certificate"]
                for effect in payload["budget"].get("systematic_effects", [])
            }
            if certificate_id in certificates:
                references.append(
                    {
                        "report": row["id"],
                        "model_id": row["model_id"],
                        "model_version": row["model_version"],
                        "supersedes": row["supersedes"],
                    }
                )
        return references
