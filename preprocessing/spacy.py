import logging
import spacy
import unicodedata
from cassis import Cas
from preprocessing.api import T_TOKEN, T_SENT, T_POS, T_DEP, T_LEMMA
from preprocessing.util import get_aslan_typesystem

logger = logging.getLogger(__name__)

# Base model names for each language (without size suffix)
# Supports all languages available in spaCy 3.8
LANGUAGE_MODELS = {
    # Web/General domain models
    "en": "en_core_web",  # English
    "zh": "zh_core_web",  # Chinese
    # News domain models
    "ca": "ca_core_news",  # Catalan
    "da": "da_core_news",  # Danish
    "de": "de_core_news",  # German
    "el": "el_core_news",  # Greek
    "es": "es_core_news",  # Spanish
    "fi": "fi_core_news",  # Finnish
    "fr": "fr_core_news",  # French
    "hr": "hr_core_news",  # Croatian
    "it": "it_core_news",  # Italian
    "ja": "ja_core_news",  # Japanese
    "ko": "ko_core_news",  # Korean
    "lt": "lt_core_news",  # Lithuanian
    "mk": "mk_core_news",  # Macedonian
    "nb": "nb_core_news",  # Norwegian Bokmål
    "nl": "nl_core_news",  # Dutch
    "pl": "pl_core_news",  # Polish
    "pt": "pt_core_news",  # Portuguese
    "ro": "ro_core_news",  # Romanian
    "ru": "ru_core_news",  # Russian
    "sl": "sl_core_news",  # Slovenian
    "sv": "sv_core_news",  # Swedish
    "uk": "uk_core_news",  # Ukrainian
}

# Default model size when not specified
DEFAULT_SIZE = "md"

# Available model sizes
VALID_SIZES = {"sm", "md", "lg"}

# Cache loaded models to avoid reloading
_MODEL_CACHE = {}


def _resolve_model_name(language: str, model_name: str | None = None, size: str | None = None) -> str:
    """
    Resolve the actual model name to load.
    
    Args:
        language: Language code (e.g., 'en', 'de')
        model_name: Custom model name (if provided, this is returned as-is)
        size: Model size: 'sm', 'md', or 'lg' (default: 'md')
    
    Returns:
        The resolved model name to load
        
    Raises:
        ValueError: If language is not supported or size is invalid
    """
    # If custom model name is provided, use it as-is
    if model_name is not None:
        return model_name
    
    # Check if language is supported
    if language not in LANGUAGE_MODELS:
        raise ValueError(
            f"Language '{language}' not supported. "
            f"Supported languages: {', '.join(LANGUAGE_MODELS.keys())}. "
            f"Or provide a custom model_name."
        )
    
    # Use default size if not specified
    if size is None:
        size = DEFAULT_SIZE
    
    # Validate size
    if size not in VALID_SIZES:
        raise ValueError(
            f"Invalid size '{size}'. Valid sizes: {', '.join(sorted(VALID_SIZES))}"
        )
    
    # Construct model name: base_name + "_" + size
    return f"{LANGUAGE_MODELS[language]}_{size}"


class Spacy_Preprocessor:
    def __init__(self, language: str, model_name: str | None = None, size: str | None = None):
        """
        Initialize the spaCy preprocessor.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'fr', 'sl')
            model_name: Custom model name. If provided, 'language' and 'size' are ignored.
                        Assumes the model is already installed.
            size: Model size ('sm', 'md', 'lg'). Default: 'md'. Ignored if model_name is provided.
        """
        self.language = language
        self.size = size or DEFAULT_SIZE
        self.model_name = _resolve_model_name(language, model_name, size)
        
        # Lazy load model on first use
        self.nlp = None
        self.ts = get_aslan_typesystem()
    
    def _load_model(self):
        """Lazy load the spacy model on first use."""
        if self.nlp is not None:
            return self.nlp
        
        # Check cache first
        if self.model_name in _MODEL_CACHE:
            self.nlp = _MODEL_CACHE[self.model_name]
            return self.nlp
        
        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            _MODEL_CACHE[self.model_name] = self.nlp
        except OSError as e:
            raise OSError(
                f"Model '{self.model_name}' not found. Install it with:\n"
                f"  pip install preprocessing[{self.language}]\n"
                f"Or download manually:\n"
                f"  python -m spacy download {self.model_name}"
            ) from e
        
        return self.nlp

    def _clean_string(self, text: str) -> str:
        ''' Remove control characters and extra whitespace '''
        cleaned = ''.join(ch for ch in text if unicodedata.category(ch) != "Cc")
        unstretched = ' '.join(cleaned.split())
        return unstretched

    def run(self, text) -> Cas:
        self.cas = Cas(self.ts)

        # converting from spaCy to DKPro is challenging
        # we discard control characters and multiple whitespaces here to make handling offsets easier
        cleaned = self._clean_string(text)
        self.cas.sofa_string = cleaned
        nlp = self._load_model()
        doc = nlp(cleaned)

        T = self.ts.get_type(T_TOKEN)
        S = self.ts.get_type(T_SENT)
        P = self.ts.get_type(T_POS)
        D = self.ts.get_type(T_DEP)
        L = self.ts.get_type(T_LEMMA)

        for sent in doc.sents:
            cas_sentence = S(begin=sent.start_char, end=sent.end_char)
            self.cas.add(cas_sentence)

        token_annos = []
        for token in doc:
            # TODO need to map from spacy pos tags to dkpro 
            cas_pos = P(begin=token.idx, end=token.idx+len(token.text), PosValue=token.tag_)
            self.cas.add(cas_pos)
            
            cas_lemma = L(begin=token.idx, end=token.idx+len(token.text), value=token.lemma_)
            self.cas.add(cas_lemma)

            cas_token = T(
                begin=token.idx, 
                end=token.idx+len(token.text), 
                id=token.i,
                pos=cas_pos,
                lemma=cas_lemma
                )
            self.cas.add(cas_token)
            token_annos.append(cas_token)

        # need another loop to ensure that all tokens are already in the CAS
        for token in doc:
            token_anno = token_annos[token.i]

            # special handling for root tokens
            governor = token_annos[token.head.i]
            if token.head == token:
                governor = token_anno

            cas_dep = D(
                begin=token_anno.begin, 
                end=token_anno.end, 
                Governor=governor,
                Dependent=token_annos[token.i], 
                DependencyType=token.dep_,
                flavor='basic'
            )
            self.cas.add(cas_dep)

        return self.cas