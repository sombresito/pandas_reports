import sys
import types
import importlib
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# stub heavy dependencies before importing rag_pipeline
model_calls = []
client_calls = []

class FakeModel:
    def __init__(self, *a, **k):
        model_calls.append('init')
    def encode(self, *a, **k):
        class Arr:
            def tolist(self):
                return [0.0]

        return Arr()

st_mod = types.ModuleType("sentence_transformers")
st_mod.SentenceTransformer = FakeModel
sys.modules['sentence_transformers'] = st_mod

class FakeHit:
    def __init__(self, text):
        self.payload = {'rag_text': text}

class FakeClient:
    def __init__(self, *a, **k):
        client_calls.append('init')
    def search(self, *a, **k):
        return [FakeHit('txt')]
    def scroll(self, *a, **k):
        return ([FakeHit('scroll')], None)

qc_mod = types.ModuleType('qdrant_client')
qc_mod.QdrantClient = FakeClient
qc_mod.QdrantConnectionError = type('QdrantConnectionError', (Exception,), {})
sys.modules['qdrant_client'] = qc_mod
sys.modules.setdefault('qdrant_client.http', types.ModuleType('qdrant_client.http'))
models_mod = types.ModuleType('qdrant_client.http.models')
models_mod.Filter = lambda *a, **k: None
models_mod.FieldCondition = lambda *a, **k: None
models_mod.MatchValue = lambda *a, **k: None
sys.modules['qdrant_client.http.models'] = models_mod
exc_mod = types.ModuleType('qdrant_client.http.exceptions')
exc_mod.UnexpectedResponse = type('UnexpectedResponse', (Exception,), {})
sys.modules['qdrant_client.http.exceptions'] = exc_mod

# requests may not be installed during tests
sys.modules.setdefault('requests', types.ModuleType('requests'))

rp = importlib.import_module("rag_pipeline")
importlib.reload(rp)
model_calls.clear()
client_calls.clear()

# replace the public name with a stub so later imports get a dummy module
stub = types.ModuleType("rag_pipeline")
stub.run_rag_analysis = lambda *a, **k: {}
stub.RagAnalysisError = type("RagAnalysisError", (Exception,), {})
sys.modules["rag_pipeline"] = stub


def test_lazy_model_and_client_creation():
    model_calls.clear()
    client_calls.clear()
    assert rp._MODEL is None
    assert rp._CLIENT is None
    assert model_calls == [] and client_calls == []

    rp.search_similar_chunks('q')
    assert model_calls == ['init']
    assert client_calls == ['init']

    # second call should not create new instances
    rp.search_similar_chunks('q2')
    assert model_calls == ['init']
    assert client_calls == ['init']


def test_run_rag_analysis_uses_cached_client(monkeypatch):
    # patch answer generation to avoid network
    monkeypatch.setattr(rp, 'generate_answer_with_ollama', lambda *a, **k: '')

    # clear calls and cache
    model_calls.clear()
    client_calls.clear()
    rp._MODEL = None
    rp._CLIENT = None

    rp.run_rag_analysis('team')
    assert model_calls == []  # model not needed
    assert client_calls == ['init']

    rp.run_rag_analysis('team')
    assert client_calls == ['init']  # cached


def test_run_rag_analysis_reraises(monkeypatch):
    monkeypatch.setattr(rp, 'generate_answer_with_ollama', lambda *a, **k: '')

    class BadClient:
        def scroll(self, *a, **k):
            raise rp.UnexpectedResponse('boom')

    monkeypatch.setattr(rp, 'get_client', lambda: BadClient())

    with pytest.raises(rp.RagAnalysisError):
        rp.run_rag_analysis('team')


def test_generate_answer_http_error(monkeypatch):
    class FakeError(Exception):
        pass

    class Resp:
        def raise_for_status(self):
            raise FakeError("boom")
        def iter_lines(self):
            return []

    req_mod = types.SimpleNamespace(post=lambda *a, **k: Resp(), RequestException=FakeError)
    monkeypatch.setattr(rp, 'requests', req_mod)

    with pytest.raises(rp.RagAnalysisError):
        rp.generate_answer_with_ollama(["c"], "q")


