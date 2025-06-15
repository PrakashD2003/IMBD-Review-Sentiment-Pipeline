import json
from src.utils.main_utils import save_json, load_json


def test_save_and_load_json(tmp_path):
    data = {"a": 1, "b": {"c": 2}}
    file_path = tmp_path / "test.json"
    save_json(str(file_path), data)
    loaded = load_json(str(file_path))
    assert loaded == data
