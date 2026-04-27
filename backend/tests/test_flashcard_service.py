from backend.services import flashcard as flashcard_service


class _FakeMeaning:
    def __init__(self, meaning: str):
        self.meaning = meaning


class _FakeWord:
    def __init__(self, word_id: int, kanji: str, kana: str, meanings: list[str]):
        self.id = word_id
        self.kanji = kanji
        self.kana = kana
        self.meanings = [_FakeMeaning(m) for m in meanings]


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        if isinstance(self._rows, list):
            return self._rows[0] if self._rows else None
        return self._rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, words: dict[int, _FakeWord], wordbank_rows: list[tuple[int]]):
        self._words = words
        self._wordbank_rows = wordbank_rows

    def query(self, column):
        if hasattr(column, "name") and column.name == "word_id":
            return _FakeQuery(self._wordbank_rows)
        return _FakeQuery(list(self._words.values()))


def test_generate_flashcard_joins_meanings(monkeypatch):
    word = _FakeWord(1, "学生", "がくせい", ["student", "pupil"])
    session = _FakeSession(words={1: word}, wordbank_rows=[])

    def _query_override(_model):
        class _WordQuery(_FakeQuery):
            def first(self_inner):
                return word

        return _WordQuery(word)

    monkeypatch.setattr(session, "query", _query_override)
    card = flashcard_service.generate_flashcard(session, 1)
    assert card.word_id == 1
    assert card.meaning == "student; pupil"


def test_format_flashcards_enrichment_flag(monkeypatch):
    monkeypatch.setattr(
        flashcard_service,
        "generate_flashcard",
        lambda _session, word_id, _show_kana: flashcard_service.Flashcard(
            word_id=word_id,
            kanji="生",
            kana="せい",
            meaning="life",
            show_kana=True,
        ),
    )
    monkeypatch.setattr(
        flashcard_service,
        "get_word_enrichment",
        lambda word_id: {"related_words": [{"id": word_id + 1}]},
    )

    payload = flashcard_service.format_flashcards(
        session=object(),
        word_ids=[1, 2],
        enrich=True,
    )
    assert "flashcards" in payload
    assert payload["flashcards"][0]["enrichment"]["related_words"][0]["id"] == 2

