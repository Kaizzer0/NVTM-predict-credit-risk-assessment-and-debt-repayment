from time import monotonic, sleep
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main, training_jobs


def test_training_token_status_and_failed_real_loader(tmp_path, monkeypatch):
    settings = SimpleNamespace(artifact_dir=tmp_path, ml_training_token="local-test-token")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(training_jobs, "get_settings", lambda: settings)
    monkeypatch.setattr(training_jobs, "_STATUS", dict(training_jobs._STATUS))
    monkeypatch.setenv("HOME_CREDIT_APPLICATION_CSV", str(tmp_path / "application_train.csv"))

    with TestClient(main.app) as client:
        assert client.get("/training").json()["status"] == "idle"
        assert client.post("/training").status_code == 403
        headers = {"X-ML-Training-Token": settings.ml_training_token}
        accepted = client.post("/training", headers=headers)
        assert accepted.status_code == 202
        assert accepted.json()["status"] == "running"
        assert client.post("/training", headers=headers).status_code == 409
        deadline = monotonic() + 15
        while monotonic() < deadline:
            complete = client.get("/training").json()
            if complete["status"] != "running":
                break
            sleep(0.1)
        assert complete["status"] == "failed"
        assert "Missing application_train.csv" in complete["message"]
        assert not (tmp_path / "active_model.json").exists()
