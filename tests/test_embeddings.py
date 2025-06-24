import sys
import types
import time
import pathlib
import importlib

# stub numpy so tests do not require the real package
np_mod = types.ModuleType("numpy")
np_mod.save = lambda path, data: open(path, "wb").write(b"x")
sys.modules.setdefault("numpy", np_mod)

# minimal pandas stub
pd_mod = types.ModuleType("pandas")
pd_mod.read_json = lambda *a, **k: None
sys.modules.setdefault("pandas", pd_mod)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# stub sentence_transformers so embeddings module can be imported without deps
model_calls = []


class FakeModel:
    def __init__(self, *a, **k):
        model_calls.append("init")

    def encode(self, texts, **k):
        return [f"e{t}" for t in texts]


st_mod = types.ModuleType("sentence_transformers")
st_mod.SentenceTransformer = FakeModel
sys.modules.setdefault("sentence_transformers", st_mod)

embeddings = importlib.import_module("embeddings")
save_embeddings = embeddings.save_embeddings


def test_embeddings_cleanup(tmp_path):
    base = tmp_path
    for i in range(5):
        save_embeddings([i], "team1", f"id{i}", base_dir=base)
        time.sleep(0.01)

    files = sorted((base / "team1").iterdir())
    assert len(files) == 3
    assert {f.name for f in files} == {"id2.npy", "id3.npy", "id4.npy"}


def test_create_embeddings_cached_model():
    class DF:
        def __init__(self, texts):
            self._texts = texts

        def __getitem__(self, key):
            class Col:
                def __init__(self, texts):
                    self._texts = texts

                def tolist(self):
                    return self._texts

            return Col(self._texts)

    model_calls.clear()
    embeddings._MODEL = None

    df = DF(["a", "b"])
    embeddings.create_embeddings(df)
    assert model_calls == ["init"]

    embeddings.create_embeddings(df)
    assert model_calls == ["init"]
