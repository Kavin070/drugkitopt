from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "app.py"
BACKUP_PATH = PROJECT_ROOT / "app" / "app.py.pre_stage8"


def test_stage8_backup_exists():
    assert BACKUP_PATH.exists()


def test_stage8_integration_exists():
    text = APP_PATH.read_text(encoding="utf-8")
    assert "# ================= STAGE 8: AGENTIC AI INTEGRATION =================" in text
    assert "Run Agentic AI Analysis" in text
    assert "DRUGKIT_API_URL" in text


def test_original_app_content_is_preserved():
    original = BACKUP_PATH.read_text(encoding="utf-8")
    current = APP_PATH.read_text(encoding="utf-8")

    # The entire previous app must remain a prefix; Stage 8 is additive.
    assert current.startswith(original)
