import numpy as np
import pytest
from segmentation import Word, Span, acoustic_bounds, padded_bounds, segment_words


def test_short_complete_utterances_with_pause_stay_separate():
    words = [Word(0, 2, 'はい。'), Word(2.8, 5.8, 'わかりました。')]
    spans = segment_words(words)
    assert [s.text for s in spans] == ['はい。', 'わかりました。']


def test_zero_duration_punctuation_preserves_sentence_boundary():
    words = [Word(0, 6, '一文目'), Word(6, 6, '。'), Word(6.1, 12.1, '二文目'), Word(12.1, 12.1, '。')]
    spans = segment_words(words)
    assert [s.text for s in spans] == ['一文目。', '二文目。']
    assert words[0].text == '一文目'


def test_long_sentence_with_short_hesitation_remains_whole():
    words = [Word(0, 6, '今日は'), Word(6.9, 14, '長い文章を話しています。')]
    spans = segment_words(words)
    assert len(spans) == 1 and spans[0].end == 14


def test_signal_refinement_recovers_early_onset_and_late_tail():
    rate = 16000
    audio = np.zeros(rate * 4, dtype=np.float32)
    audio[int(.75 * rate):int(2.25 * rate)] = .2
    spans = [Span(1, 2, 'テスト。', 1)]
    start, end = acoustic_bounds(spans, audio, rate)[0]
    assert start <= .75 and end >= 2.25
    assert 0 <= start < spans[0].start < spans[0].end < end <= 4


def test_end_recovers_half_second_tail_and_keeps_quiet_padding():
    rate = 16000
    audio = np.zeros(rate * 5, dtype=np.float32)
    audio[rate:int(2.55 * rate)] = .2
    span = Span(1, 2, '語尾が伸びます。', 1)
    _, end = acoustic_bounds([span], audio, rate)[0]
    assert 2.65 <= end <= 2.70


def test_end_skips_brief_dip_and_preserves_soft_fading_vowel():
    rate = 16000
    audio = np.zeros(rate * 5, dtype=np.float32)
    audio[rate:int(2.30 * rate)] = .2
    audio[int(2.04 * rate):int(2.09 * rate)] = 0
    # Quiet enough to fool the old 8% threshold, still audible as a vowel tail.
    audio[int(2.30 * rate):int(2.58 * rate)] = .01
    _, end = acoustic_bounds([Span(1, 2, 'テスト。', 1)], audio, rate)[0]
    assert end >= 2.68


def test_extended_end_stays_before_next_clip_and_source_end():
    rate = 16000
    audio = np.zeros(rate * 4, dtype=np.float32)
    audio[rate:int(2.5 * rate)] = .2
    audio[int(2.85 * rate):int(3.5 * rate)] = .2
    spans = [Span(1, 2.1, '一文目。', 1), Span(2.85, 3.5, '二文目。', 1)]
    bounds = acoustic_bounds(spans, audio, rate)
    assert bounds[0][1] <= bounds[1][0] and bounds[-1][1] <= 4


def test_start_removes_clearly_silent_early_timestamp_with_preroll():
    rate = 16000
    audio = np.zeros(rate * 4, dtype=np.float32)
    audio[int(1.50 * rate):int(2.5 * rate)] = .2
    start, end = acoustic_bounds([Span(1, 2.5, 'テスト。', 1)], audio, rate)[0]
    assert 1.37 <= start <= 1.39 and end >= 2.5


def test_start_keeps_soft_onset_and_does_not_skip_long_unknown_region():
    rate = 16000
    audio = np.zeros(rate * 4, dtype=np.float32)
    audio[rate:int(1.5 * rate)] = .01
    audio[int(1.5 * rate):int(2.5 * rate)] = .2
    assert acoustic_bounds([Span(1, 2.5, 'テスト。', 1)], audio, rate)[0][0] <= 1
    audio[:int(1.9 * rate)] = 0
    assert acoustic_bounds([Span(1, 2.5, 'テスト。', 1)], audio, rate)[0][0] <= 1


def test_noise_without_silence_falls_back_to_padding():
    audio = np.full(48000 * 5, .1, dtype=np.float32)
    spans = [Span(1, 2, 'テスト。', 1)]
    assert acoustic_bounds(spans, audio, 48000) == padded_bounds(spans, 5)


def test_adjacent_clips_never_overlap_or_include_neighbor_words():
    rate = 16000
    audio = np.zeros(rate * 4, dtype=np.float32)
    audio[rate:2 * rate] = .1
    audio[int(2.1 * rate):3 * rate] = .1
    spans = [Span(1, 2, '一文目。', 1), Span(2.1, 3, '二文目。', 1)]
    bounds = acoustic_bounds(spans, audio, rate)
    assert bounds[0][1] <= bounds[1][0]
    for (start, end), span in zip(bounds, spans):
        assert start <= span.start and end >= span.end
    assert bounds[0][1] < spans[1].start and bounds[1][0] > spans[0].end


@pytest.mark.parametrize('audio', [np.array([np.nan]), np.zeros((2, 5))])
def test_invalid_signal(audio):
    with pytest.raises(ValueError):
        acoustic_bounds([], audio, 16000)


def test_fast_intro_missing_punctuation_with_final_question_splits():
    words = [Word(0, 8, '初めましてかなたです'), Word(8.3, 17, '歌うことが好きです'),
             Word(17.25, 26, '曲も作っています'), Word(26.3, 35, '聴いてくださいね'),
             Word(35.25, 44, '応援してください'), Word(44.3, 51, '一緒に歩んでくれますか?')]
    spans = segment_words(words)
    assert len(spans) == 6
    assert ''.join(s.text for s in spans) == ''.join(w.text for w in words)
    assert all(s.end-s.start <= 10 for s in spans)
    assert '要試聴' in spans[0].note


def test_ordinary_short_material_keeps_original_grouping():
    words = [Word(0, 6, '今日はいい天気ですね'), Word(6.3, 13.8, '一緒に出かけませんか?')]
    spans = segment_words(words)
    assert len(spans) == 1 and spans[0].note == ''


def test_long_actual_sentence_not_cut_only_for_duration():
    words = [Word(0, 9, '今日は'), Word(9.3, 18, '皆さんのために'), Word(18.3, 30, 'とても長い一文を話しています。')]
    spans = segment_words(words)
    assert len(spans) == 1 and spans[0].end == 30


def test_conjunctive_particle_after_polite_ending_is_not_sentence_end():
    words = [Word(0, 8, '好きです'), Word(8.3, 9, 'が'), Word(9.3, 30, 'まだ練習している途中です。')]
    assert len(segment_words(words)) == 1
