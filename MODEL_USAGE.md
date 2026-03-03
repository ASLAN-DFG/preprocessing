# Model Name Resolution - Usage Examples

This document demonstrates the new model name resolution system for the spaCy preprocessor.

## Overview

The new system supports three ways to specify models:

1. **Language only** → Uses medium model (default)
2. **Language + size** → Uses specified size (sm/md/lg)
3. **Custom model name** → Uses custom model as-is

## Usage Examples

### 1. Minimal Usage (defaults to medium model)

```python
from preprocessing.spacy import Spacy_Preprocessor

# English with default medium model (en_core_web_md)
preprocessor = Spacy_Preprocessor(language='en')

# German with default medium model (de_core_news_md)
preprocessor = Spacy_Preprocessor(language='de')
```

### 2. Specifying Model Size

```python
# Small English model
preprocessor = Spacy_Preprocessor(language='en', size='sm')  # en_core_web_sm

# Large German model
preprocessor = Spacy_Preprocessor(language='de', size='lg')  # de_core_news_lg

# Medium French model (explicit)
preprocessor = Spacy_Preprocessor(language='fr', size='md')  # fr_core_news_md
```

### 3. Custom Model Name

```python
# Use a custom model (assumes it's already installed)
preprocessor = Spacy_Preprocessor(
    language='en',  # language parameter is required
    model_name='my_custom_spacy_model'
)
# When model_name is provided, 'size' is ignored
```

## Supported Languages (24 total)

| Language | Code | Base Model Name | Domain |
|----------|------|-----------------|--------|
| Catalan | `ca` | `ca_core_news` | News |
| Chinese | `zh` | `zh_core_web` | Web |
| Croatian | `hr` | `hr_core_news` | News |
| Danish | `da` | `da_core_news` | News |
| Dutch | `nl` | `nl_core_news` | News |
| English | `en` | `en_core_web` | Web |
| Finnish | `fi` | `fi_core_news` | News |
| French | `fr` | `fr_core_news` | News |
| German | `de` | `de_core_news` | News |
| Greek | `el` | `el_core_news` | News |
| Italian | `it` | `it_core_news` | News |
| Japanese | `ja` | `ja_core_news` | News |
| Korean | `ko` | `ko_core_news` | News |
| Lithuanian | `lt` | `lt_core_news` | News |
| Macedonian | `mk` | `mk_core_news` | News |
| Norwegian Bokmål | `nb` | `nb_core_news` | News |
| Polish | `pl` | `pl_core_news` | News |
| Portuguese | `pt` | `pt_core_news` | News |
| Romanian | `ro` | `ro_core_news` | News |
| Russian | `ru` | `ru_core_news` | News |
| Slovenian | `sl` | `sl_core_news` | News |
| Spanish | `es` | `es_core_news` | News |
| Swedish | `sv` | `sv_core_news` | News |
| Ukrainian | `uk` | `uk_core_news` | News |

**Note:** English and Chinese use the 'web' domain, while all other languages use the 'news' domain.

## Supported Model Sizes

- `sm` - Small model
- `md` - Medium model (default)
- `lg` - Large model

## Installation

### Install all models for a specific language:

```bash
# Install all English models (sm, md, lg)
pip install preprocessing[en]

# Install all German models
pip install preprocessing[de]

# Install all Japanese models
pip install preprocessing[ja]

# Install models for multiple languages
pip install preprocessing[en,de,fr,pt]
```

### Install all models for all languages:

```bash
pip install preprocessing[all]
```

### Or install specific models manually:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_md
python -m spacy download zh_core_web_lg
python -m spacy download ja_core_news_md
```

## Error Handling

### Invalid language:
```python
# This will raise ValueError
preprocessor = Spacy_Preprocessor(language='xx')
# Error: Language 'xx' not supported. Supported languages: ca, zh, hr, da, ...
```

### Invalid size:
```python
# This will raise ValueError
preprocessor = Spacy_Preprocessor(language='en', size='xl')
# Error: Invalid size 'xl'. Valid sizes: lg, md, sm
```

### Model not found:
```python
# This will raise OSError at model load time
preprocessor = Spacy_Preprocessor(language='en')
text = preprocessor.run("test")  # OSError if model not installed
```

## Model Resolution Logic

The `_resolve_model_name()` function uses the following logic:

1. If `model_name` is provided → return it unchanged
2. Check if `language` is supported → raise error if not
3. Use `size` parameter if provided, otherwise default to `'md'`
4. Validate `size` is one of {`sm`, `md`, `lg`} → raise error if not
5. Construct model name: `{base_model}_{size}`

## Examples

```python
# All of these load the same model (en_core_web_md):
Spacy_Preprocessor(language='en')
Spacy_Preprocessor(language='en', size='md')
Spacy_Preprocessor(language='en', model_name='en_core_web_md')

# Different sizes:
Spacy_Preprocessor(language='en', size='sm')  # en_core_web_sm
Spacy_Preprocessor(language='en', size='lg')  # en_core_web_lg

# Different languages:
Spacy_Preprocessor(language='de', size='md')  # de_core_news_md
Spacy_Preprocessor(language='ja', size='sm')  # ja_core_news_sm
Spacy_Preprocessor(language='pt', size='lg')  # pt_core_news_lg

# Custom model:
Spacy_Preprocessor(language='en', model_name='my_model')  # my_model
```

## Special Cases

### Floret Vectors

The following languages use **floret vectors** instead of default word vectors in their md and lg models:
- Croatian (hr)
- Finnish (fi)
- Korean (ko)
- Slovenian (sl)
- Swedish (sv)
- Ukrainian (uk)

This improves handling of out-of-vocabulary (OOV) tokens, but affects some vector operations. See the [spaCy documentation](https://spacy.io/usage/v3-2#vectors) for more details.

### Domain Information

- **Web domain** models: Trained on web text (blogs, news, comments)
  - English (`en_core_web`)
  - Chinese (`zh_core_web`)

- **News domain** models: Trained on news text
  - All other 22 languages
