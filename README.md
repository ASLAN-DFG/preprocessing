# Preprocessing

Preprocessing utilities for linguistic data with lazy-loaded spaCy models.

## Installation

### Basic installation (without language models)

```bash
pip install preprocessing
```

Models will be downloaded on first use. You'll need to specify which language model to install.

### With language models

Install with specific language models:

```bash
# Single language
pip install "preprocessing[en]"       # English
pip install "preprocessing[de]"       # German
pip install "preprocessing[ja]"       # Japanese
pip install "preprocessing[pt]"       # Portuguese

# Multiple languages
pip install "preprocessing[en,de,fr,pt,ja]"

# All 24 languages
pip install "preprocessing[all]"
```

### From source with Poetry

```bash
poetry install
```

Or with specific language models:

```bash
poetry install --with en,de,ja
poetry install --with all  # All 24 languages
```

## Usage

```python
from preprocessing.spacy import Spacy_Preprocessor

# Automatically uses default medium model for any supported language
processor = Spacy_Preprocessor("en")  # Uses en_core_web_md
processor = Spacy_Preprocessor("de")  # Uses de_core_news_md
processor = Spacy_Preprocessor("ja")  # Uses ja_core_news_md

# Specify model size: sm (small), md (medium, default), lg (large)
processor = Spacy_Preprocessor("en", size="sm")   # en_core_web_sm
processor = Spacy_Preprocessor("de", size="lg")   # de_core_news_lg

# Or specify a custom model
processor = Spacy_Preprocessor("en", model_name="en_core_web_sm")

# Model is loaded lazily on first use
cas = processor.run("Your text here")
```

## Supported Languages (24 total)

**Web domain:**
- **en** - English
- **zh** - Chinese

**News domain:**
- **ca** - Catalan
- **da** - Danish
- **de** - German
- **el** - Greek
- **es** - Spanish
- **fi** - Finnish
- **fr** - French
- **hr** - Croatian
- **it** - Italian
- **ja** - Japanese
- **ko** - Korean
- **lt** - Lithuanian
- **mk** - Macedonian
- **nb** - Norwegian Bokmål
- **nl** - Dutch
- **pl** - Polish
- **pt** - Portuguese
- **ro** - Romanian
- **ru** - Russian
- **sl** - Slovenian
- **sv** - Swedish
- **uk** - Ukrainian

Each language supports three model sizes: `sm` (small), `md` (medium, default), and `lg` (large).

For detailed usage examples, see [MODEL_USAGE.md](MODEL_USAGE.md).

## Development

Install development dependencies:

```bash
poetry install --with dev
```

Run tests:

```bash
poetry run pytest
```
