import os
import time
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import types

pandas_chunking_stub = types.ModuleType("pandas_chunking")
pandas_chunking_stub.chunk_json_to_jsonl = lambda *a, **k: None
sys.modules.setdefault("pandas_chunking", pandas_chunking_stub)

rag_pipeline_stub = types.ModuleType("rag_pipeline")
rag_pipeline_stub.run_rag_analysis = lambda *a, **k: {}
sys.modules.setdefault("rag_pipeline", rag_pipeline_stub)

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)

import utils


def _fake_chunk_json_to_jsonl(data, path, uuid):
    with open(path, "w") as f:
        f.write(uuid)


def test_cleanup_keeps_three_most_recent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "chunk_json_to_jsonl", _fake_chunk_json_to_jsonl)

    team = "team1"
    for i in range(5):
        utils.chunk_and_save_json({}, f"id{i}", team)
        time.sleep(0.01)

    files = sorted(os.listdir(tmp_path / "chunks" / team))
    assert len(files) == 3
    assert set(files) == {"id2.jsonl", "id3.jsonl", "id4.jsonl"}


def test_no_deletion_when_three_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "chunk_json_to_jsonl", _fake_chunk_json_to_jsonl)

    team = "team2"
    for i in range(3):
        utils.chunk_and_save_json({}, f"id{i}", team)
        time.sleep(0.01)

    files = sorted(os.listdir(tmp_path / "chunks" / team))
    assert len(files) == 3
    assert set(files) == {"id0.jsonl", "id1.jsonl", "id2.jsonl"}
