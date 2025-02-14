import os
import pytest
from app.src.main import startup


@pytest.fixture(scope="session", autouse=True)
def load_model():
    startup()
    yield
