from irodori_tts.text_normalization import normalize_text


def test_normalize_text_applies_product_pronunciation_hints():
    assert normalize_text("GensparkとClaudeを使います。") == "ジェン・スパークとクロードを使います。"
    assert normalize_text("ｇｅｎｓｐａｒｋとclaude") == "ジェン・スパークとクロード"


def test_normalize_text_removes_spaces_between_japanese_text_units():
    assert normalize_text("クリック したくなるタイトルを10案出して というプロンプト") == (
        "クリックしたくなるタイトルを10案出してというプロンプト"
    )
