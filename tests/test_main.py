import sys
import types
import asyncio
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# --- provide minimal fastapi, requests and dotenv stubs ---
fastapi_mod = types.ModuleType("fastapi")

requests_mod = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_mod)

pd_mod = types.ModuleType("pandas")
pd_mod.DataFrame = object
sys.modules.setdefault("pandas", pd_mod)

pandas_chunking_stub = types.ModuleType("pandas_chunking")
pandas_chunking_stub.chunk_json_to_jsonl = lambda *a, **k: None
sys.modules.setdefault("pandas_chunking", pandas_chunking_stub)

rag_pipeline_stub = types.ModuleType("rag_pipeline")
rag_pipeline_stub.run_rag_analysis = lambda *a, **k: {}
rag_pipeline_stub.RagAnalysisError = type("RagAnalysisError", (Exception,), {})
sys.modules.setdefault("rag_pipeline", rag_pipeline_stub)

class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

class Request:
    def __init__(self, json_data):
        self._json = json_data
    async def json(self):
        return self._json

class FastAPI:
    def __init__(self):
        self.routes = {}
    def post(self, path):
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._json = body
    def json(self):
        return self._json

class TestClient:
    def __init__(self, app):
        self.app = app
    def post(self, path, json=None):
        handler = self.app.routes[path]
        req = Request(json)
        try:
            if asyncio.iscoroutinefunction(handler):
                body = asyncio.get_event_loop().run_until_complete(handler(req))
            else:
                body = handler(req)
            status = 200
        except HTTPException as exc:
            body = {"detail": exc.detail}
            status = exc.status_code
        return _Resp(status, body)

testclient_mod = types.ModuleType("fastapi.testclient")
testclient_mod.TestClient = TestClient

fastapi_mod.FastAPI = FastAPI
fastapi_mod.HTTPException = HTTPException
fastapi_mod.Request = Request
fastapi_mod.testclient = testclient_mod
sys.modules.setdefault("fastapi", fastapi_mod)
sys.modules.setdefault("fastapi.testclient", testclient_mod)

dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dotenv_mod)

import main


def _setup_success(monkeypatch):
    def fake_get(url, timeout, **kwargs):
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {}
        return Resp()

    monkeypatch.setattr(main, "ALLURE_API", "http://example")
    monkeypatch.setattr(main.requests, "get", fake_get, raising=False)
    monkeypatch.setattr(main, "extract_team_name", lambda data: "team1")
    monkeypatch.setattr(main, "chunk_and_save_json", lambda *a, **k: None)


def test_analyze_success(monkeypatch):
    _setup_success(monkeypatch)
    called = {}
    def fake_analyze(uuid, team):
        called['val'] = (uuid, team)
    monkeypatch.setattr(main, "analyze_and_post", fake_analyze)

    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    resp = client.post("/uuid/analyze", json={"uuid": "id1"})
    assert resp.status_code == 200
    assert resp.json() == {"result": "ok", "team": "team1"}
    assert called['val'] == ("id1", "team1")


def test_missing_uuid(monkeypatch):
    _setup_success(monkeypatch)
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    resp = client.post("/uuid/analyze", json={})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "UUID not provided."


def test_partial_on_error(monkeypatch):
    _setup_success(monkeypatch)
    def fail(uuid, team):
        raise RuntimeError("boom")
    monkeypatch.setattr(main, "analyze_and_post", fail)

    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    resp = client.post("/uuid/analyze", json={"uuid": "id1"})
    assert resp.status_code == 200
    assert resp.json() == {"result": "partial", "error": "boom"}
