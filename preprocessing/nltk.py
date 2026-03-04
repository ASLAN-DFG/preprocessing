import logging
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from cassis import Cas

from preprocessing.api import BasePreprocessor, T_TOKEN, T_SENT, T_POS, T_DEP, T_LEMMA

logger = logging.getLogger(__name__)

# Default language for NLTK (English only)
DEFAULT_LANGUAGE = "en"

# Mapping from Penn Treebank POS tags to Universal POS tags (UPOS) for consistency
PENN_TO_UPOS = {
    # Nouns
    "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
    # Verbs
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
    # Adjectives
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    # Adverbs
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    # Determiners
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    # Pronouns
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON",
    # Prepositions
    "IN": "ADP",
    # Conjunctions
    "CC": "CONJ", "CD": "NUM",
    # Other
    "RP": "PRT",
    "UH": "INTJ",
    "SYM": "SYM",
    "FW": "X",
    ".": "PUNCT", ",": "PUNCT", ":": "PUNCT", "-LRB-": "PUNCT", "-RRB-": "PUNCT",
    "''": "PUNCT", "``": "PUNCT", "-": "PUNCT",
}

# Cache loaded resources to avoid reloading
_NLTK_CACHE = {}

# Required NLTK resources
REQUIRED_RESOURCES = ["punkt", "averaged_perceptron_tagger", "wordnet", "universal_tagmap"]


def _ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded."""
    for resource in REQUIRED_RESOURCES:
        try:
            # Try to load the resource
            if resource == "punkt":
                nltk.data.find(f"tokenizers/{resource}")
            elif resource == "averaged_perceptron_tagger":
                nltk.data.find(f"taggers/{resource}")
            elif resource == "wordnet":
                nltk.data.find(f"corpora/{resource}")
            elif resource == "universal_tagmap":
                nltk.data.find(f"taggers/{resource}")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")


def _get_character_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """
    Calculate character offsets for each token in the text.
    
    Args:
        text: The original text
        tokens: List of token strings
        
    Returns:
        List of (begin, end) tuples for each token
    """
    offsets = []
    current_pos = 0
    
    for token in tokens:
        # Find the token starting from current position
        idx = text.find(token, current_pos)
        if idx == -1:
            # Token not found (shouldn't happen with proper tokenization)
            logger.warning(f"Token '{token}' not found in text starting from position {current_pos}")
            offsets.append((current_pos, current_pos))
            continue
        
        offsets.append((idx, idx + len(token)))
        current_pos = idx + len(token)
    
    return offsets


class NLTK_Preprocessor(BasePreprocessor):
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        """
        Initialize the NLTK preprocessor.
        
        Args:
            language: Language code. Currently only 'en' (English) is supported.
                      
        Raises:
            ValueError: If language is not 'en'
        """
        if language != "en":
            raise ValueError(
                f"NLTK preprocessor currently supports English only. "
                f"Requested language: '{language}'"
            )
        
        super().__init__(language)
        
        # Ensure NLTK resources are available
        _ensure_nltk_resources()
        
        # Lazy load resources on first use
        self.lemmatizer = None
    
    def _load_lemmatizer(self):
        """Lazy load the WordNetLemmatizer on first use."""
        if self.lemmatizer is not None:
            return self.lemmatizer
        
        # Check cache first
        if "wordnet_lemmatizer" in _NLTK_CACHE:
            self.lemmatizer = _NLTK_CACHE["wordnet_lemmatizer"]
            return self.lemmatizer
        
        logger.info("Loading NLTK WordNetLemmatizer")
        self.lemmatizer = WordNetLemmatizer()
        _NLTK_CACHE["wordnet_lemmatizer"] = self.lemmatizer
        
        return self.lemmatizer
    
    def _convert_penntree_to_wordnet_pos(self, penn_pos: str) -> str:
        """Convert Penn Treebank POS tag to WordNet POS tag for lemmatization.
        
        Args:
            penn_pos: Penn Treebank POS tag (e.g., 'NN', 'VBZ')
            
        Returns:
            WordNet POS tag ('n', 'v', 'a', 'r') or 'n' as default
        """
        # Simplified mapping for lemmatization
        if penn_pos.startswith('V'):
            return 'v'  # Verb
        elif penn_pos.startswith('JJ'):
            return 'a'  # Adjective
        elif penn_pos.startswith('RB'):
            return 'r'  # Adverb
        else:
            return 'n'  # Default to noun
    
    def _penn_to_upos(self, penn_pos: str) -> str:
        """Convert Penn Treebank POS tag to Universal POS tag.
        
        Args:
            penn_pos: Penn Treebank POS tag
            
        Returns:
            Universal POS tag (UPOS)
        """
        return PENN_TO_UPOS.get(penn_pos, "X")
    
    def _extract_sentences_with_offsets(self, text: str) -> list[tuple[int, int]]:
        """Extract sentences with character offsets.
        
        Args:
            text: The input text
            
        Returns:
            List of (begin, end) tuples for each sentence
        """
        sentences = sent_tokenize(text)
        offsets = []
        current_pos = 0
        
        for sent in sentences:
            # Find the sentence starting from current position
            idx = text.find(sent, current_pos)
            if idx == -1:
                logger.warning(f"Sentence not found in text: {sent}")
                continue
            
            offsets.append((idx, idx + len(sent)))
            current_pos = idx + len(sent)
        
        return offsets

    def run(self, text: str) -> Cas:
        """
        Process text and return a CASSIS CAS object with annotations.
        
        Args:
            text: Input text to process
            
        Returns:
            Cas: CASSIS CAS object with extracted annotations
        """
        self.cas = Cas(self.ts)
        
        # Clean and normalize text
        cleaned = self._clean_string(text)
        self.cas.sofa_string = cleaned
        
        # Load lemmatizer
        lemmatizer = self._load_lemmatizer()
        
        # Tokenize sentences
        sentence_offsets = self._extract_sentences_with_offsets(cleaned)
        
        # Get type handles for annotations
        T = self.ts.get_type(T_TOKEN)
        S = self.ts.get_type(T_SENT)
        P = self.ts.get_type(T_POS)
        D = self.ts.get_type(T_DEP)
        L = self.ts.get_type(T_LEMMA)
        
        # Add sentence annotations
        for sent_begin, sent_end in sentence_offsets:
            cas_sentence = S(begin=sent_begin, end=sent_end)
            self.cas.add(cas_sentence)
        
        # Tokenize and POS tag the entire text
        tokens = word_tokenize(cleaned)
        pos_tags = pos_tag(tokens)
        
        # Get character offsets for all tokens
        token_offsets = _get_character_offsets(cleaned, tokens)
        
        # Second pass: add tokens and their annotations
        token_annos = []  # Track token annotations for dependency linking
        
        for token_id, (token_text, penn_pos) in enumerate(pos_tags):
            begin, end = token_offsets[token_id]
            
            # Convert Penn Treebank POS to UPOS
            upos = self._penn_to_upos(penn_pos)
            
            # Create POS annotation
            cas_pos = P(begin=begin, end=end, PosValue=upos)
            self.cas.add(cas_pos)
            
            # Lemmatize using WordNet (convert POS tag for better lemmatization)
            wordnet_pos = self._convert_penntree_to_wordnet_pos(penn_pos)
            lemma_text = lemmatizer.lemmatize(token_text.lower(), pos=wordnet_pos)
            
            # Create lemma annotation
            cas_lemma = L(begin=begin, end=end, value=lemma_text)
            self.cas.add(cas_lemma)
            
            # Create token annotation
            cas_token = T(
                begin=begin,
                end=end,
                id=token_id,
                pos=cas_pos,
                lemma=cas_lemma
            )
            self.cas.add(cas_token)
            token_annos.append(cas_token)
        
        # Try to add dependency relations (NLTK has limited support)
        # For now, we'll use a simple approach: create self-relations for all tokens
        # This is a fallback; real dependency parsing would require external tools
        try:
            for token_id, cas_token in enumerate(token_annos):
                # Simplified approach: each token depends on itself (ROOT)
                # or use the first pass as a dummy dependency structure
                # This is a limitation of NLTK
                
                # For now, we'll create a simple dependency structure
                # where all tokens depend on the root token (index 0)
                if token_id == 0:
                    governor = cas_token
                    dep_type = "ROOT"
                else:
                    governor = token_annos[0]
                    dep_type = "dep"
                
                cas_dep = D(
                    begin=cas_token.begin,
                    end=cas_token.end,
                    Governor=governor,
                    Dependent=cas_token,
                    DependencyType=dep_type,
                    flavor="basic"
                )
                self.cas.add(cas_dep)
        except Exception as e:
            logger.warning(f"Failed to add dependency relations: {e}")
        
        return self.cas
