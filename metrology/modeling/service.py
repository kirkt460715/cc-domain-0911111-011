"""Application facade for versioned measurement-model uncertainty work."""

from ..errors import InputError
from ..values import identifier, instant
from .evaluation import ModelEvaluation
from .store import ModelStore


class ModelService:
    def __init__(self, database, catalog):
        self.store = ModelStore(database, catalog)
        self.evaluation = ModelEvaluation(database, catalog)
        self.db = database
        self.catalog = catalog

    def register_model(self, data, *, created_at):
        return self.store.register_model(data, created_at)

    def model(self, model_id, version=None):
        return self.store.get_model(model_id, version).as_dict()

    def models(self):
        return self.store.list_models()

    def evaluate(
        self,
        model_id,
        request,
        *,
        report_id,
        created_at,
        version=None,
        supersedes=None,
        correction_reason=None,
    ):
        identifier(report_id, "report_id")
        timestamp = instant(created_at)
        if supersedes is not None:
            identifier(supersedes, "supersedes")
        model = self.store.get_model(model_id, version)
        report = self.evaluation.evaluate(
            model,
            request,
            report_id=report_id,
            created_at=timestamp,
            supersedes=supersedes,
        )
        if correction_reason is not None:
            if not isinstance(correction_reason, str):
                raise InputError("correction reason must be text")
            report["correction_reason"] = correction_reason
            report["fingerprint"] = self._fingerprint(report)
        self.store.save_report(report, created_at=timestamp)
        return report

    @staticmethod
    def _fingerprint(report):
        from ..values import digest

        return digest(
            {key: value for key, value in report.items() if key != "fingerprint"}
        )

    def report(self, report_id):
        return self.store.get_report(report_id)

    def reports(self, model_id=None):
        return self.store.list_reports(model_id)

    def certificate_reports(self, certificate_id):
        identifier(certificate_id)
        return self.store.reports_for_certificate(certificate_id)

    def corrected_report(
        self,
        model_id,
        request,
        *,
        report_id,
        supersedes,
        created_at,
        version=None,
        reason=None,
    ):
        """Issue a separate report after a certificate correction.

        The previous report is never modified; the new report records the
        superseded report id and the correction reason.
        """

        return self.evaluate(
            model_id,
            request,
            report_id=report_id,
            created_at=created_at,
            version=version,
            supersedes=supersedes,
            correction_reason=reason,
        )
