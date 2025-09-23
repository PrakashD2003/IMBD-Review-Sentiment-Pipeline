import pytest
from common.utils.main_utils import save_json, load_json


def test_save_and_load_json(tmp_path):
    data = {"a": 1, "b": {"c": 2}}
    file_path = tmp_path / "test.json"
    save_json(str(file_path), data)
    loaded = load_json(str(file_path))
    assert loaded == data

def test_load_params_file_not_found(tmp_path):
    """
    Tests that load_params raises FileNotFoundError for a non-existent file.
    """
    non_existent_path = tmp_path / "non_existent_params.yaml"
    
    with pytest.raises(FileNotFoundError):
        load_params(str(non_existent_path))