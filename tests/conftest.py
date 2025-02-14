import os
import pytest
from app.src.main import startup


@pytest.fixture(scope="session", autouse=True)
def load_model():
    current_directory = os.getcwd()
    print("---------------")
    print(f"Текущая директория: {current_directory}")

    # Список файлов и папок в текущей директории
    files_and_folders = os.listdir(current_directory)
    print("Файлы и папки в текущей директории:")
    for item in files_and_folders:
        print(item)
    startup()
    yield
