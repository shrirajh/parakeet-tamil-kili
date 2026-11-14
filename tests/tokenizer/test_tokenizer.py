"""ta-rom romanisation (pure Python) and the tokenizer merge (needs sentencepiece; skips without it)."""
import pytest

from kili.tokenizer.romanize import deromanize, romanize


@pytest.mark.parametrize("ta", ["தமிழ்", "க்கு", "நாளைக்கு", "வீட்லலாம்", "பள்ளி", "மழை", "ஃபோன்", "கஇ", "ஸ்ரீ", "அன்னம்"])
def test_romanize_round_trip(ta):
    rom = romanize(ta)
    assert rom.isascii()
    assert deromanize(rom) == ta


def test_romanize_passes_latin_through():
    assert romanize("நாளைக்கு meeting 5 மணி") == f"{romanize('நாளைக்கு')} meeting 5 {romanize('மணி')}"


def test_warm_start_maps_a_tamil_piece_to_its_sound():
    assert romanize("க்கு") == "kku"


def test_merge_appends_tamil_after_v3(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    from kili.tokenizer.extend import merge, train_ta_bpe
    en = tmp_path / "en.txt"
    en.write_text("\n".join(["the quick brown fox jumps over the lazy dog"] * 200), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(en), model_prefix=str(tmp_path / "base"), vocab_size=40, model_type="bpe", minloglevel=2)
    ta = tmp_path / "ta.txt"
    ta.write_text("\n".join(["நாளைக்கு மீட்டிங்க்கு வருவியா", "தமிழ் பேசுவோம்", "எங்க வீட்டுக்கு வாங்க"] * 300), encoding="utf-8")
    ta_model = train_ta_bpe(str(ta), 60, str(tmp_path / "ta"))
    base = spm.SentencePieceProcessor(model_file=str(tmp_path / "base.model"))
    proto, added = merge(base.serialized_model_proto(), ta_model)
    new = spm.SentencePieceProcessor(model_proto=proto)
    assert added and all(any("஀" <= c <= "௿" for c in p) for p in added)
    assert new.get_piece_size() == base.get_piece_size() + len(added)
    assert all(new.id_to_piece(i) == base.id_to_piece(i) for i in range(base.get_piece_size()))  # old ids unchanged
    assert new.encode("the lazy dog") == base.encode("the lazy dog")
