import pytest
from cassis import Cas
from preprocessing.api import T_TOKEN, T_SENT, T_POS, T_DEP, T_LEMMA
from preprocessing.util import get_aslan_typesystem
from preprocessing.spacy import Spacy_Preprocessor

def test_spacy_preprocessing():
    ts = get_aslan_typesystem()
    text = "Demokratie ist eher abstrakt. Leben ist konkret."
    spacy = Spacy_Preprocessor(language='de')
    cas = spacy.run(text)

    gold_tokens = ["Demokratie", "ist", "eher", "abstrakt", ".", "Leben", "ist", "konkret", "."]
    gold_lemmas = ["Demokratie", "sein", "eher", "abstrakt", ".", "Leben", "sein", "konkret", "."]
    gold_pos = ["NN", "VAFIN", "ADV", "ADJD", "$.", "NN", "VAFIN", "ADJD", "$."]
    gold_sentences = [[0, 29], [30, 48]]
    gold_tok_index = [[0, 10], [11, 14], [15, 19], [20, 28], [28, 29], [30, 35], [36, 39], [40, 47], [47, 48]]
    gold_dep = [
                [gold_tokens[1], gold_tokens[0], 'nsubj', 'basic'],
                [gold_tokens[1], gold_tokens[1], 'ROOT', 'basic'],
                [gold_tokens[3], gold_tokens[2], 'cop', 'basic'],
                [gold_tokens[1], gold_tokens[3], 'advmod', 'basic'],
                [gold_tokens[1], gold_tokens[4], 'punct', 'basic'],
                [gold_tokens[6], gold_tokens[5], 'nsubj', 'basic'],
                [gold_tokens[6], gold_tokens[6], 'ROOT', 'basic'],
                [gold_tokens[6], gold_tokens[7], 'cop', 'basic'],
                [gold_tokens[6], gold_tokens[8], 'punct', 'basic']]
    
    assert all([a.get_covered_text() == b for a, b in zip(cas.select(T_TOKEN), gold_tokens)])
    assert all([a.get('PosValue') == b for a, b in zip(cas.select(T_POS), gold_pos)])

    found_sents = []
    for sent in cas.select(T_SENT):
        print([sent.begin, sent.end])
        found_sents.append([sent.begin, sent.end])
    for i in range(len(found_sents)):
        assert found_sents[i] == gold_sentences[i]



    # TODO really check all deps
    found_dep = []
    for dep in cas.select(T_DEP):
        print(dep)
        found_dep.append([cas.sofa_string[dep.Governor.begin: dep.Governor.end],
                                 cas.sofa_string[dep.Dependent.begin: dep.Dependent.end],
                                 #TODO spacy dependencytype mapping to dkpro dependencytype
                                 #dep.DependencyType,
                                 dep.flavor])
        #assert dep.get('Dependent') is not None
        #assert dep.get('Governor') is not None
    for i in range(len(found_dep)):
        assert found_dep[i] == [gold_dep[i][0], gold_dep[i][1], gold_dep[i][3]]

def test_spacy_prepro_en():
    text = 'This is a test. A small one.'
    ts = get_aslan_typesystem()
    spacy = Spacy_Preprocessor(language='en')
    cas = spacy.run(text)

    expected_token = [[0,4], [5,7], [8,9], [10,14], [14,15], [16,17], [18,23], [24,27], [27,28]]
    expected_sentences = [[0,15], [16,28]]
    expected_pos = [
        ['DT', 0, 4],
        ['VBZ', 5, 7],
        ['DT', 8, 9],
        ['NN', 10, 14],
        ['.', 14, 15],
        ['DT', 16, 17],
        ['JJ', 18, 23],
        ['NN', 24, 27],
        ['.', 27, 28]]
    expected_dependencies = [
        ['is', 'This', 'nsubj', 'basic'],
        ['is', 'is', 'ROOT', 'basic'],
        ['test', 'a', 'det', 'basic'],
        ['is', 'test','attr','basic'],
        ['is', '.', 'punct', 'basic'],
        ['one', 'A', 'det', 'basic'],
        ['one', 'small', 'amod', 'basic'],
        ['one', 'one', 'ROOT', 'basic'],
        ['one', '.', 'punct', 'basic']]
    actual_token = []
    actual_sentences = []
    actual_pos = []
    actual_dependencies = []

    for token in cas.select(T_TOKEN):
        actual_token.append([token.begin, token.end])
    for sent in cas.select(T_SENT):
        actual_sentences.append([sent.begin, sent.end])
    for pos in cas.select(T_POS):
        actual_pos.append([pos.PosValue, pos.begin, pos.end])
    for dep in cas.select(T_DEP):
        print(dep)
        actual_dependencies.append([dep.Governor.get_covered_text(), dep.Dependent.get_covered_text(), dep.DependencyType, dep.flavor])

    assert len(expected_token) == len(actual_token)
    assert len(expected_sentences) == len(actual_sentences)
    assert len(expected_pos) == len(actual_pos)
    assert len(expected_dependencies) == len(actual_dependencies)

    for i in range(len(actual_token)):
        assert expected_token[i] == actual_token[i]
    for i in range(len(actual_sentences)):
        assert expected_sentences[i] == actual_sentences[i]
    for i in range(len(actual_pos)):
        assert expected_pos[i] == actual_pos[i]
    for i in range(len(actual_dependencies)):
        assert expected_dependencies[i] == actual_dependencies[i]

def test_lazy_model_loading():
    """Test that spaCy models are loaded lazily and cached."""
    from preprocessing.spacy import _MODEL_CACHE
    
    # Clear cache before test
    _MODEL_CACHE.clear()
    
    # 1. Test that model is not loaded on instantiation
    spacy_preprocessor = Spacy_Preprocessor(language='en')
    assert spacy_preprocessor.nlp is None, "Model should not be loaded on instantiation"
    
    # 2. Test that model is loaded on first use
    text = "This is a test."
    cas = spacy_preprocessor.run(text)
    assert spacy_preprocessor.nlp is not None, "Model should be loaded after first use"
    first_nlp = spacy_preprocessor.nlp
    
    # 3. Test that cache is used for subsequent instantiations
    spacy_preprocessor2 = Spacy_Preprocessor(language='en')
    assert spacy_preprocessor2.nlp is None, "Second instance should also have lazy loading"
    
    cas2 = spacy_preprocessor2.run(text)
    assert spacy_preprocessor2.nlp is first_nlp, "Second instance should use cached model"
    
    # 4. Test cache contains the model
    assert 'en_core_web_md' in _MODEL_CACHE, "Model should be in cache"
    assert _MODEL_CACHE['en_core_web_md'] is first_nlp, "Cached model should be the same instance"