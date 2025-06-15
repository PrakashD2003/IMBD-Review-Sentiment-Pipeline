from src.data.data_preprocessing import preprocess_text


def test_preprocess_text_basic():
    text = "This is a TEST! Visit https://example.com for more info. 100% great"
    cleaned = preprocess_text(text)
    assert cleaned == "test visit info great"