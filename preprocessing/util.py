import cassis
import pathlib
from functools import lru_cache
from cassis import Cas
from preprocessing.api import T_TOKEN, T_LEMMA, T_POS
from typing import Type, List, Union, Optional
from importlib.resources import files


# --- Singleton TypeSystem ------------------------------------------------

@lru_cache(maxsize=1)
def get_aslan_typesystem() -> cassis.TypeSystem:
    """Return a singleton TypeSystem instance.
    This guarantees identity equality across the process.
    """
    with (files("data") / "TypeSystem.xml").open("rb") as f:
        ts = cassis.load_typesystem(f)
    return ts