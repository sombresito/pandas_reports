import sys, types, importlib, pathlib

# ensure repo root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# minimal numpy and pandas stubs
np_mod = sys.modules.setdefault("numpy", types.ModuleType("numpy"))
class FakeArray:
    def __init__(self, arr):
        self._arr = arr
        self.shape = (len(arr), len(arr[0]) if arr else 0)
    def __getitem__(self, idx):
        return self._arr[idx]
np_mod.array = lambda a: FakeArray(a)
np_mod.ndarray = FakeArray

pd_mod = sys.modules.setdefault("pandas", types.ModuleType("pandas"))
pd_mod.DataFrame = object

# sentence_transformers stub so embeddings imports succeed
st_mod = types.ModuleType("sentence_transformers")
class FakeModel:
    def __init__(self, *a, **k):
        pass
    def encode(self, *a, **k):
        class Arr:
            def tolist(self):
                return [0.0]

        return Arr()
st_mod.SentenceTransformer = FakeModel
sys.modules.setdefault("sentence_transformers", st_mod)

# stub qdrant client and models
qc_mod = types.ModuleType("qdrant_client")
class _QC:
    def __init__(self, *a, **k):
        pass
qc_mod.QdrantClient = _QC
sys.modules.setdefault("qdrant_client", qc_mod)
sys.modules.setdefault("qdrant_client.http", types.ModuleType("qdrant_client.http"))
models_mod = sys.modules.setdefault(
    "qdrant_client.http.models", types.ModuleType("qdrant_client.http.models")
)
models_mod.Distance = getattr(models_mod, "Distance", object)
models_mod.VectorParams = getattr(models_mod, "VectorParams", object)
class PointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.payload = payload
models_mod.PointStruct = PointStruct
models_mod.Filter = getattr(models_mod, "Filter", lambda *a, **k: None)
models_mod.FieldCondition = getattr(models_mod, "FieldCondition", lambda *a, **k: None)
models_mod.MatchValue = getattr(models_mod, "MatchValue", lambda *a, **k: None)
class PointIdsList:
    def __init__(self, points):
        self.points = points
models_mod.PointIdsList = PointIdsList

se = importlib.import_module("save_embeddings_to_qdrant")
importlib.reload(se)

class Point:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload

class FakeClient:
    def __init__(self):
        self.deleted = []
        self.upserted = []
    def collection_exists(self, collection_name):
        return True
    def create_collection(self, *a, **k):
        pass
    def scroll(self, *a, **k):
        return ([
            Point(1, {"report_uuid": "old1"}),
            Point(2, {"report_uuid": "old1"}),
            Point(3, {"report_uuid": "old2"}),
            Point(4, {"report_uuid": "old3"}),
        ], None)
    def delete(self, collection_name, points_selector):
        self.deleted.append(points_selector.points)
    def upsert(self, collection_name, points):
        self.upserted.extend(points)

class DF:
    def __init__(self, row):
        self._row = row
    def __len__(self):
        return 1
    class _Loc:
        def __init__(self, row):
            self._row = row
        def __getitem__(self, key):
            idx, col = key
            return self._row[col]
    @property
    def loc(self):
        return DF._Loc(self._row)


def test_upload_embeddings_deletes_old_reports():
    df = DF({"rag_text": "t", "name": "n", "status": "s", "suite": "su", "uid": "u"})
    embeddings = se.np.array([[0.0]])
    client = FakeClient()
    se.upload_embeddings(df, embeddings, "team", "new", client=client)
    assert client.deleted == [[1, 2]]
