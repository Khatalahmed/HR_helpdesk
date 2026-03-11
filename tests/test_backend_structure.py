from pathlib import Path


def test_backend_files_exist():
    expected = [
        "app/main.py",
        "app/config.py",
        "app/schemas.py",
        "app/rag/router.py",
        "app/rag/service.py",
        "app/rag/prompt.py",
    ]
    for rel_path in expected:
        assert Path(rel_path).exists(), f"Missing file: {rel_path}"

