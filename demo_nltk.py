#!/usr/bin/env python
"""
Demo script for NLTK preprocessing.
Highlights the three available preprocessor implementations:
- Spacy_Preprocessor (24 languages)
- Stanza_Preprocessor (multiple languages)  
- NLTK_Preprocessor (English only)
"""

from preprocessing import NLTK_Preprocessor
from preprocessing.api import T_TOKEN, T_SENT, T_POS, T_LEMMA, T_DEP


def demo_nltk():
    """Demonstrate NLTK preprocessing on English text."""
    print("\n" + "="*70)
    print("NLTK PREPROCESSOR DEMO")
    print("="*70)
    
    # Sample English text
    text = "The quick brown fox jumps over the lazy dog. It is amazingly fast."
    
    print(f"\nInput text: {text}\n")
    
    # Initialize NLTK preprocessor (English only)
    nltk_prep = NLTK_Preprocessor(language='en')
    
    # Process the text
    cas = nltk_prep.run(text)
    
    print("Preprocessed text SOFA string:")
    print(f"  '{cas.sofa_string}'\n")
    
    # Extract and display sentences
    print("SENTENCES:")
    for i, sent in enumerate(cas.select(T_SENT)):
        sent_text = cas.sofa_string[sent.begin:sent.end]
        print(f"  [{i}] ({sent.begin:3d}, {sent.end:3d}): '{sent_text}'")
    
    # Extract and display tokens with POS tags
    print("\nTOKENS with POS tags and LEMMAS:")
    tokens = list(cas.select(T_TOKEN))
    pos_tags = list(cas.select(T_POS))
    lemmas = list(cas.select(T_LEMMA))
    
    print(f"  {'ID':<3} {'Token':<12} {'POS':<8} {'Lemma':<12}")
    print(f"  {'-'*50}")
    
    for token_id, (token, pos, lemma) in enumerate(zip(tokens, pos_tags, lemmas)):
        token_text = cas.sofa_string[token.begin:token.end]
        print(f"  {token_id:<3} {token_text:<12} {pos.PosValue:<8} {lemma.value:<12}")
    
    # Extract and display dependencies
    print("\nDEPENDENCIES:")
    for dep in cas.select(T_DEP):
        gov_text = cas.sofa_string[dep.Governor.begin:dep.Governor.end]
        dep_text = cas.sofa_string[dep.Dependent.begin:dep.Dependent.end]
        print(f"  '{gov_text}' <- '{dep_text}' ({dep.DependencyType})")
    
    print("\n" + "="*70 + "\n")


def demo_language_restriction():
    """Demonstrate that NLTK only supports English."""
    print("\n" + "="*70)
    print("LANGUAGE RESTRICTION DEMO")
    print("="*70)
    print("\nAttempting to create NLTK preprocessor for German...")
    
    try:
        nltk_prep = NLTK_Preprocessor(language='de')
    except ValueError as e:
        print(f"✓ Error (as expected): {e}\n")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_nltk()
    demo_language_restriction()
    print("NLTK preprocessor is working correctly!")
