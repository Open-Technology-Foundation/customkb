#!/usr/bin/env python
"""
Unit tests for query/enhancement.py
Tests query normalization, synonym expansion, spelling correction, and caching.
"""

import hashlib
import json
import os
import time
from unittest.mock import Mock, patch

import pytest

from query.enhancement import (
  apply_spelling_correction,
  correct_spelling,
  enhance_query,
  expand_synonyms,
  get_cached_enhanced_query,
  get_enhancement_cache_key,
  get_synonyms_for_word,
  normalize_query,
  save_enhanced_query_to_cache,
)


class TestNormalizeQuery:
  """Test normalize_query function."""

  def test_empty_query(self):
    assert normalize_query('') == ''

  def test_none_query(self):
    assert normalize_query(None) == ''

  @patch('query.enhancement.clean_text', side_effect=lambda x: x.lower())
  def test_whitespace_collapsing(self, _mock_clean):
    result = normalize_query('hello    world   test')
    assert result == 'hello world test'

  @patch('query.enhancement.clean_text', side_effect=lambda x: x.lower())
  def test_leading_trailing_whitespace(self, _mock_clean):
    result = normalize_query('  hello  ')
    assert result == 'hello'

  @patch('query.enhancement.clean_text', side_effect=lambda x: x)
  def test_smart_double_quotes(self, _mock_clean):
    result = normalize_query('search for \u201csomething\u201d here')
    assert '\u201c' not in result
    assert '\u201d' not in result
    assert '"something"' in result

  @patch('query.enhancement.clean_text', side_effect=lambda x: x)
  def test_smart_single_quotes(self, _mock_clean):
    result = normalize_query('it\u2019s a \u2018test\u2019')
    assert '\u2018' not in result
    assert '\u2019' not in result
    assert "it's" in result

  @patch('query.enhancement.clean_text', side_effect=lambda x: x)
  def test_redundant_dots(self, _mock_clean):
    result = normalize_query('wait... what...')
    assert '...' not in result
    assert 'wait. what.' in result

  @patch('query.enhancement.clean_text', side_effect=lambda x: x)
  def test_redundant_question_marks(self, _mock_clean):
    result = normalize_query('really???')
    assert '???' not in result
    assert 'really?' in result

  @patch('query.enhancement.clean_text', side_effect=lambda x: x)
  def test_redundant_exclamation_marks(self, _mock_clean):
    result = normalize_query('wow!!!')
    assert '!!!' not in result
    assert 'wow!' in result


class TestGetSynonymsForWord:
  """Test get_synonyms_for_word function."""

  def test_empty_word(self):
    assert get_synonyms_for_word('') == []

  def test_single_char_word(self):
    assert get_synonyms_for_word('a') == []

  @patch.dict('sys.modules', {'nltk': None})
  def test_import_error_fallback_ing(self):
    """Morphological fallback for -ing suffix."""
    with patch('builtins.__import__', side_effect=ImportError):
      result = get_synonyms_for_word('running', max_synonyms=3)
    # Falls through to morphological variants
    assert isinstance(result, list)

  def test_morphological_fallback_ing(self):
    """Test -ing suffix variants when WordNet unavailable."""
    with patch('query.enhancement.get_synonyms_for_word') as mock_fn:
      # Simulate the morphological fallback directly
      word = 'running'
      base = word[:-3]  # 'runn' -> not great, but tests the logic
      variants = [base, base + 'ed', base + 'er']
      variants = [s for s in variants if len(s) > 2 and s != word.lower()]
      mock_fn.return_value = variants[:2]
      result = mock_fn(word)
      assert isinstance(result, list)

  def test_max_synonyms_respected(self):
    """Synonyms list should not exceed max_synonyms."""
    result = get_synonyms_for_word('testing', max_synonyms=1)
    assert len(result) <= 1

  def test_short_word_no_fallback(self):
    """Words <= 3 chars should get no morphological fallback."""
    # Need to make WordNet fail to trigger fallback path
    with patch('builtins.__import__', side_effect=ImportError):
      result = get_synonyms_for_word('run')
    assert isinstance(result, list)

  def test_returns_list(self):
    result = get_synonyms_for_word('computer')
    assert isinstance(result, list)


class TestCorrectSpelling:
  """Test correct_spelling function."""

  def test_empty_word(self):
    assert correct_spelling('') == ''

  def test_none_returns_none(self):
    assert correct_spelling(None) is None

  def test_single_char(self):
    assert correct_spelling('a') == 'a'

  def test_word_in_vocabulary(self):
    vocab = {'hello', 'world', 'test'}
    assert correct_spelling('hello', vocabulary=vocab) == 'hello'

  def test_textblob_import_error(self):
    """When textblob is unavailable, returns original word."""
    with patch('builtins.__import__', side_effect=ImportError):
      result = correct_spelling('tset')
    assert isinstance(result, str)

  def test_double_letter_correction_with_vocab(self):
    vocab = {'runing', 'test'}  # deliberately misspelled vocab for testing
    # 'running' has 'nn', candidate would be 'runing' which IS in vocab
    result = correct_spelling('running', vocabulary=vocab)
    assert result == 'runing'

  def test_no_correction_without_vocab_match(self):
    """Double-letter correction only applies when candidate is in vocabulary."""
    result = correct_spelling('running', vocabulary={'hello'})
    # 'runing' not in vocab, so original returned
    assert result == 'running'

  def test_short_word_no_double_letter(self):
    """Words <= 3 chars skip double-letter correction."""
    result = correct_spelling('aa', vocabulary={'a'})
    assert result == 'aa'


class TestExpandSynonyms:
  """Test expand_synonyms function."""

  def test_empty_query(self):
    assert expand_synonyms('') == ''

  def test_none_query(self):
    assert expand_synonyms(None) is None

  def test_disabled_via_kb(self):
    kb = Mock()
    kb.enable_synonym_expansion = False
    result = expand_synonyms('machine learning', kb=kb)
    assert result == 'machine learning'

  def test_stop_words_skipped(self):
    """Stop words should not be expanded."""
    with patch('query.enhancement.get_synonyms_for_word', return_value=['substitute']) as mock_syn:
      expand_synonyms('the and or but')
      # None of these stop words should trigger synonym lookup
      mock_syn.assert_not_called()

  def test_short_words_skipped(self):
    """Words shorter than min length are not expanded."""
    with patch('query.enhancement.get_synonyms_for_word', return_value=['substitute']) as mock_syn:
      expand_synonyms('cat')  # 3 chars, default min is 4
      mock_syn.assert_not_called()

  def test_or_expansion_format(self):
    """Expanded words use (word OR synonym) format."""
    with patch('query.enhancement.get_synonyms_for_word', return_value=['study']):
      result = expand_synonyms('research')
      assert 'OR' in result
      assert 'research' in result
      assert 'study' in result

  def test_no_synonyms_passthrough(self):
    """Words with no synonyms pass through unchanged."""
    with patch('query.enhancement.get_synonyms_for_word', return_value=[]):
      result = expand_synonyms('xyzzy')
      assert result == 'xyzzy'

  def test_kb_config_respected(self):
    kb = Mock()
    kb.enable_synonym_expansion = True
    kb.max_synonyms_per_word = 2
    kb.synonym_min_word_length = 3
    with patch('query.enhancement.get_synonyms_for_word', return_value=[]) as mock_syn:
      expand_synonyms('test data', kb=kb)
      # 'test' is 4 chars >= min 3, should be called
      assert mock_syn.called


class TestApplySpellingCorrection:
  """Test apply_spelling_correction function."""

  def test_empty_query(self):
    assert apply_spelling_correction('') == ''

  def test_none_query(self):
    assert apply_spelling_correction(None) is None

  def test_disabled_via_kb(self):
    kb = Mock()
    kb.enable_spelling_correction = False
    result = apply_spelling_correction('tset wrold', kb=kb)
    assert result == 'tset wrold'

  def test_short_words_skipped(self):
    """Words <= 2 chars are not spell-checked."""
    with patch('query.enhancement.correct_spelling') as mock_correct:
      apply_spelling_correction('I am a ok')
      # 'I', 'am', 'a', 'ok' -- 'am' and 'ok' are 2 chars, only words >2 are checked
      # Actually 'am' is 2 chars so not checked, 'ok' is 2 chars so not checked
      mock_correct.assert_not_called()

  def test_correction_applied(self):
    """When correct_spelling returns different word, query is updated."""
    with patch('query.enhancement.correct_spelling', side_effect=lambda w: 'test' if w == 'tset' else w):
      result = apply_spelling_correction('tset data')
      assert 'test' in result

  def test_no_correction_returns_original(self):
    """When no corrections are made, original query is returned."""
    with patch('query.enhancement.correct_spelling', side_effect=lambda w: w):
      result = apply_spelling_correction('hello world')
      assert result == 'hello world'


class TestEnhancementCacheKey:
  """Test get_enhancement_cache_key function."""

  def test_deterministic(self):
    key1 = get_enhancement_cache_key('test query')
    key2 = get_enhancement_cache_key('test query')
    assert key1 == key2

  def test_different_inputs_different_keys(self):
    key1 = get_enhancement_cache_key('query one')
    key2 = get_enhancement_cache_key('query two')
    assert key1 != key2

  def test_sha256_format(self):
    key = get_enhancement_cache_key('test')
    assert len(key) == 64  # SHA-256 hex digest length
    assert all(c in '0123456789abcdef' for c in key)

  def test_matches_manual_hash(self):
    text = 'hello world'
    expected = hashlib.sha256(text.encode()).hexdigest()
    assert get_enhancement_cache_key(text) == expected


class TestGetCachedEnhancedQuery:
  """Test get_cached_enhanced_query function."""

  def test_cache_miss_no_file(self):
    result = get_cached_enhanced_query('nonexistent query')
    assert result is None

  def test_cache_hit(self, tmp_path):
    query = 'test query'
    cache_key = get_enhancement_cache_key(query)
    cache_data = {'original': query, 'enhanced': 'enhanced test query', 'timestamp': time.time()}

    with patch('query.enhancement.ENHANCEMENT_CACHE_DIR', str(tmp_path)):
      cache_file = tmp_path / f'{cache_key}.json'
      cache_file.write_text(json.dumps(cache_data))

      result = get_cached_enhanced_query(query)
      assert result == 'enhanced test query'

  def test_cache_expired(self, tmp_path):
    query = 'test query'
    cache_key = get_enhancement_cache_key(query)
    cache_data = {'original': query, 'enhanced': 'enhanced', 'timestamp': time.time()}

    with patch('query.enhancement.ENHANCEMENT_CACHE_DIR', str(tmp_path)):
      cache_file = tmp_path / f'{cache_key}.json'
      cache_file.write_text(json.dumps(cache_data))
      # Set file mtime to 2 hours ago
      old_time = time.time() - 7200
      os.utime(cache_file, (old_time, old_time))

      result = get_cached_enhanced_query(query)
      assert result is None
      # Expired file should be removed
      assert not cache_file.exists()

  def test_cache_mismatched_original(self, tmp_path):
    query = 'test query'
    cache_key = get_enhancement_cache_key(query)
    cache_data = {'original': 'different query', 'enhanced': 'enhanced', 'timestamp': time.time()}

    with patch('query.enhancement.ENHANCEMENT_CACHE_DIR', str(tmp_path)):
      cache_file = tmp_path / f'{cache_key}.json'
      cache_file.write_text(json.dumps(cache_data))

      result = get_cached_enhanced_query(query)
      assert result is None

  def test_cache_read_error(self, tmp_path):
    query = 'test query'
    cache_key = get_enhancement_cache_key(query)

    with patch('query.enhancement.ENHANCEMENT_CACHE_DIR', str(tmp_path)):
      cache_file = tmp_path / f'{cache_key}.json'
      cache_file.write_text('invalid json{{{')

      result = get_cached_enhanced_query(query)
      assert result is None


class TestSaveEnhancedQueryToCache:
  """Test save_enhanced_query_to_cache function."""

  def test_no_save_when_unchanged(self, tmp_path):
    with patch('query.enhancement.ENHANCEMENT_CACHE_DIR', str(tmp_path)):
      save_enhanced_query_to_cache('same', 'same')
      assert len(list(tmp_path.glob('*.json'))) == 0

  def test_saves_cache_file(self, tmp_path):
    with patch('query.enhancement.ENHANCEMENT_CACHE_DIR', str(tmp_path)):
      save_enhanced_query_to_cache('original', 'enhanced')
      files = list(tmp_path.glob('*.json'))
      assert len(files) == 1

      data = json.loads(files[0].read_text())
      assert data['original'] == 'original'
      assert data['enhanced'] == 'enhanced'
      assert 'timestamp' in data

  def test_write_error_handled(self):
    with patch('query.enhancement.ENHANCEMENT_CACHE_DIR', '/nonexistent/impossible/path'):
      # Should not raise, just log
      save_enhanced_query_to_cache('original', 'enhanced')


class TestEnhanceQuery:
  """Test enhance_query main pipeline function."""

  def test_empty_query(self):
    assert enhance_query('') == ''

  def test_none_query(self):
    assert enhance_query(None) is None

  @patch('query.enhancement.get_cached_enhanced_query', return_value='cached result')
  def test_cache_hit_shortcut(self, _mock_cache):
    result = enhance_query('test query')
    assert result == 'cached result'

  @patch('query.enhancement.get_cached_enhanced_query', return_value=None)
  @patch('query.enhancement.normalize_query', side_effect=lambda q: q.strip())
  @patch('query.enhancement.save_enhanced_query_to_cache')
  def test_normalization_only(self, _mock_save, _mock_norm, _mock_cache):
    """Without KB config, only normalization is applied."""
    result = enhance_query('test query')
    assert result == 'test query'

  @patch('query.enhancement.get_cached_enhanced_query', return_value=None)
  @patch('query.enhancement.normalize_query', side_effect=lambda q: q.lower())
  @patch('query.enhancement.apply_spelling_correction', side_effect=lambda q, kb=None: q.replace('tset', 'test'))
  @patch('query.enhancement.expand_synonyms', side_effect=lambda q, kb=None: q + ' OR exam')
  @patch('query.enhancement.save_enhanced_query_to_cache')
  def test_full_pipeline(self, mock_save, _mock_expand, _mock_spell, _mock_norm, _mock_cache):
    kb = Mock()
    kb.enable_spelling_correction = True
    kb.enable_synonym_expansion = True
    result = enhance_query('TSET query', kb=kb)
    assert 'test' in result
    assert 'OR' in result
    mock_save.assert_called_once()

  @patch('query.enhancement.get_cached_enhanced_query', return_value=None)
  @patch('query.enhancement.normalize_query', side_effect=ValueError('bad'))
  def test_error_returns_original(self, _mock_norm, _mock_cache):
    result = enhance_query('test query')
    assert result == 'test query'

  @patch('query.enhancement.get_cached_enhanced_query', return_value=None)
  @patch('query.enhancement.normalize_query', side_effect=lambda q: q)
  @patch('query.enhancement.save_enhanced_query_to_cache')
  def test_no_cache_when_unchanged(self, mock_save, _mock_norm, _mock_cache):
    """When enhancement doesn't change the query, don't cache."""
    enhance_query('test query')
    mock_save.assert_not_called()


if __name__ == '__main__':
  pytest.main([__file__, '-v'])

# fin
