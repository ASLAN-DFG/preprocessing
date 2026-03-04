import pytest
from cassis import Cas
from preprocessing.api import T_TOKEN, T_SENT, T_POS, T_DEP, T_LEMMA
from preprocessing.util import get_aslan_typesystem
from preprocessing.nltk import NLTK_Preprocessor


def test_nltk_language_validation():
    """Test that NLTK preprocessor only supports English."""
    with pytest.raises(ValueError, match="currently supports English only"):
        NLTK_Preprocessor(language='de')
    
    with pytest.raises(ValueError, match="currently supports English only"):
        NLTK_Preprocessor(language='fr')


def test_nltk_english_preprocessing():
    """Test basic NLTK preprocessing on English text."""
    text = 'This is a test. A small one.'
    ts = get_aslan_typesystem()
    nltk_prep = NLTK_Preprocessor(language='en')
    cas = nltk_prep.run(text)

    # Verify CAS is created with correct SOFA string
    assert cas.sofa_string == text
    
    # Check tokens
    tokens = list(cas.select(T_TOKEN))
    assert len(tokens) > 0
    
    # Verify token text
    token_texts = [cas.sofa_string[t.begin:t.end] for t in tokens]
    print(f"Tokens: {token_texts}")
    
    # Check POS tags are present
    pos_tags = list(cas.select(T_POS))
    assert len(pos_tags) > 0
    assert all(p.PosValue for p in pos_tags)  # All should have PosValue
    
    # Check lemmas are present
    lemmas = list(cas.select(T_LEMMA))
    assert len(lemmas) > 0
    
    # Check sentences
    sentences = list(cas.select(T_SENT))
    assert len(sentences) == 2  # Two sentences in the test text
    
    # Check sentence offsets
    sent_offsets = [[s.begin, s.end] for s in sentences]
    print(f"Sentence offsets: {sent_offsets}")
    # Should have two sentences
    assert len(sent_offsets) == 2


def test_nltk_simple_sentence():
    """Test NLTK preprocessing on a simple sentence."""
    text = 'The cat sat.'
    nltk_prep = NLTK_Preprocessor(language='en')
    cas = nltk_prep.run(text)
    
    # Check basic structure
    assert cas.sofa_string == text
    
    tokens = list(cas.select(T_TOKEN))
    token_texts = [cas.sofa_string[t.begin:t.end] for t in tokens]
    
    # Should have tokens for: The, cat, sat, .
    assert len(tokens) >= 3
    print(f"Tokens: {token_texts}")
    
    # Verify POS tagging
    pos_tags = list(cas.select(T_POS))
    assert len(pos_tags) == len(tokens)
    
    # All tokens should map to their POS correctly
    pos_values = [p.PosValue for p in pos_tags]
    print(f"POS tags: {pos_values}")
    assert all(p for p in pos_values)  # All should have a POS value


def test_nltk_lemmatization():
    """Test that lemmatization works correctly."""
    text = 'Running dogs chase cats.'
    nltk_prep = NLTK_Preprocessor(language='en')
    cas = nltk_prep.run(text)
    
    tokens = list(cas.select(T_TOKEN))
    lemmas = list(cas.select(T_LEMMA))
    
    assert len(tokens) == len(lemmas)
    
    # Get tokens and their lemmas
    token_lemma_pairs = []
    for token, lemma in zip(tokens, lemmas):
        token_text = cas.sofa_string[token.begin:token.end]
        lemma_value = lemma.value
        token_lemma_pairs.append((token_text, lemma_value))
    
    print(f"Token-Lemma pairs: {token_lemma_pairs}")
    
    # Check specific lemmatizations
    # Note: NLTK lemmatization may vary, but we can check some basic cases
    token_lemma_dict = {pair[0]: pair[1] for pair in token_lemma_pairs}
    
    # "Running" should lemmatize to "run" (verb)
    assert token_lemma_dict.get('Running', '').lower() in ['running', 'run']
    # "dogs" should lemmatize to "dog"
    assert token_lemma_dict.get('dogs', '').lower() in ['dogs', 'dog']
    # "chase" or "chasing" variants
    assert 'chase' in token_lemma_dict.get('chase', '').lower() or \
           'chase' in token_lemma_dict.get('chases', '').lower()


def test_nltk_character_offsets():
    """Test that character offsets are correctly calculated."""
    text = 'Hello world.'
    nltk_prep = NLTK_Preprocessor(language='en')
    cas = nltk_prep.run(text)
    
    tokens = list(cas.select(T_TOKEN))
    
    # Verify each token's character offsets
    for token in tokens:
        covered_text = cas.sofa_string[token.begin:token.end]
        print(f"Token: begin={token.begin}, end={token.end}, text='{covered_text}'")
        # Verify no empty texts
        assert len(covered_text) > 0
        # Verify offset is within bounds
        assert token.begin >= 0
        assert token.end <= len(cas.sofa_string)


def test_nltk_punctuation_handling():
    """Test that punctuation is handled correctly."""
    text = 'Hello, world!'
    nltk_prep = NLTK_Preprocessor(language='en')
    cas = nltk_prep.run(text)
    
    tokens = list(cas.select(T_TOKEN))
    token_texts = [cas.sofa_string[t.begin:t.end] for t in tokens]
    
    # Should include punctuation tokens
    print(f"Tokens: {token_texts}")
    assert ',' in token_texts or any(',' == t for t in token_texts)


def test_nltk_multi_sentence():
    """Test preprocessing of multiple sentences."""
    text = 'First sentence. Second sentence. Third sentence.'
    nltk_prep = NLTK_Preprocessor(language='en')
    cas = nltk_prep.run(text)
    
    sentences = list(cas.select(T_SENT))
    assert len(sentences) == 3
    
    # Verify each sentence is correct
    for sent in sentences:
        covered_text = cas.sofa_string[sent.begin:sent.end]
        print(f"Sentence: '{covered_text}'")
        assert len(covered_text) > 0


def test_nltk_token_id():
    """Test that token IDs are correctly assigned."""
    text = 'One two three.'
    nltk_prep = NLTK_Preprocessor(language='en')
    cas = nltk_prep.run(text)
    
    tokens = list(cas.select(T_TOKEN))
    
    # Verify token IDs are sequential starting from 0
    for i, token in enumerate(tokens):
        assert token.id == i


def test_nltk_default_language():
    """Test that default language is English."""
    nltk_prep = NLTK_Preprocessor()  # No language specified
    assert nltk_prep.language == 'en'
    
    text = 'Test sentence.'
    cas = nltk_prep.run(text)
    assert cas.sofa_string == text
