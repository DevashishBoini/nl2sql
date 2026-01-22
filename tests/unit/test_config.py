import pytest
from backend.config import get_settings
from backend.constants import LogLevel


## test for import and loading settings
def test_get_settings():
    settings = get_settings()
    assert settings is not None
    assert settings.retrieval.top_k > 0 
    assert settings.llm.temperature >= 0.0
    assert settings.app.log_level in LogLevel  

## test for singleton
def test_get_settings_singleton():
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2  

