import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SETTINGS_SOURCE = Path(__file__).resolve().parents[1] / "src" / "emergency_dispatcher" / "settings.py"


def test_project_dotenv_loads_outside_cwd_and_preserves_nonempty_shell_values(tmp_path):
    project_root = tmp_path / "project"
    package_dir = project_root / "src" / "emergency_dispatcher"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(SETTINGS_SOURCE, package_dir / "settings.py")
    (project_root / ".env").write_text(
        "\n".join(
            (
                "GRADIUM_API_KEY=dotenv-gradium",
                "AI_COUSTICS_API_KEY=dotenv-acoustics",
                "OPENAI_API_KEY=dotenv-openai",
                "OPENAI_MODEL=dotenv-model",
                "",
            )
        ),
        encoding="utf-8",
    )
    different_cwd = tmp_path / "unrelated-working-directory"
    different_cwd.mkdir()

    child_env = os.environ.copy()
    for name in ("GRADIUM_API_KEY", "AI_COUSTICS_API_KEY", "OPENAI_API_KEY", "OPENAI_MODEL"):
        child_env.pop(name, None)
    child_env["GRADIUM_API_KEY"] = "shell-gradium"
    child_env["AI_COUSTICS_API_KEY"] = ""
    child_env["PYTHONPATH"] = str(project_root / "src")
    probe = """
import json
import sys
from pathlib import Path
from emergency_dispatcher.settings import REPO_ROOT, get_settings

settings = get_settings()
print(json.dumps({
    "project_root": REPO_ROOT == Path(sys.argv[1]),
    "different_cwd": Path.cwd() == Path(sys.argv[2]),
    "shell_value_preserved": settings.gradium_api_key == "shell-gradium",
    "empty_shell_value_filled": settings.ai_coustics_api_key == "dotenv-acoustics",
    "dotenv_openai_key_loaded": settings.openai_api_key == "dotenv-openai",
    "dotenv_model_loaded": settings.openai_model == "dotenv-model",
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe, str(project_root), str(different_cwd)],
        cwd=different_cwd,
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "project_root": True,
        "different_cwd": True,
        "shell_value_preserved": True,
        "empty_shell_value_filled": True,
        "dotenv_openai_key_loaded": True,
        "dotenv_model_loaded": True,
    }
