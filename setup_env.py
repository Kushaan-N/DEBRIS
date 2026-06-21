# -*- coding: utf-8 -*-
"""
setup_env.py - DEBRIS project environment setup checker

Run:  python setup_env.py
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent

# Variable definitions: (name, description/source, include_in_dotenv, default_value)
VARS = [
    (
        "HUD_API_KEY",
        "get it from: https://hud.ai/project/api-keys",
        True,
        "",
    ),
    (
        "MODAL_TOKEN_ID",
        "run `modal token new` in terminal",
        True,
        "",
    ),
    (
        "MODAL_TOKEN_SECRET",
        "run `modal token new` in terminal",
        True,
        "",
    ),
    (
        "HF_TOKEN",
        "get it from: https://huggingface.co/settings/tokens",
        True,
        "",
    ),
    (
        "HF_REPO",
        "set to your HuggingFace username/repo-name e.g. kushaan/debris-dataset",
        True,
        "",
    ),
    (
        "RECORD_DIR",
        "local path for recording datasets, suggest ./data",
        True,
        "./data",
    ),
    (
        "WORLDSIM_VIEWER",
        "set to 1 to open live 3D viewer, optional",
        False,   # not written to .env (optional, never set in Modal)
        "",
    ),
]


def mask(value: str) -> str:
    if len(value) <= 4:
        return "****"
    return value[:4] + "*" * min(len(value) - 4, 8)


def main():
    print()
    print("DEBRIS — Environment Variable Status")
    print("=" * 52)

    set_vars: set[str] = set()
    for name, source, _, _ in VARS:
        val = os.environ.get(name, "")
        if val:
            set_vars.add(name)
            print(f"  [SET]     {name} = {mask(val)}")
        else:
            print(f"  [MISSING] {name} — {source}")

    print()

    # ── .env handling ────────────────────────────────────────────────────────
    dotenv_path = ROOT / ".env"

    # Read existing .env keys so we don't overwrite them
    existing_keys: set[str] = set()
    existing_lines: list[str] = []
    if dotenv_path.exists():
        with open(dotenv_path) as f:
            existing_lines = f.readlines()
        for line in existing_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                existing_keys.add(stripped.split("=", 1)[0].strip())

    # Build new lines to append for missing vars
    new_lines: list[str] = []
    for name, _, include_in_dotenv, default_value in VARS:
        if not include_in_dotenv:
            continue
        if name in existing_keys:
            continue   # already present — don't touch it
        new_lines.append(f"{name}={default_value}\n")

    if not dotenv_path.exists():
        with open(dotenv_path, "w") as f:
            f.writelines(new_lines)
        print(f"  Created .env with {len(new_lines)} placeholder(s).")
    elif new_lines:
        with open(dotenv_path, "a") as f:
            if existing_lines and not existing_lines[-1].endswith("\n"):
                f.write("\n")
            f.writelines(new_lines)
        print(f"  Updated .env — added {len(new_lines)} missing placeholder(s).")
    else:
        print("  .env already contains all required variables — nothing changed.")

    # ── .gitignore handling ──────────────────────────────────────────────────
    gitignore_path = ROOT / ".gitignore"
    gitignore_entry = ".env\n"

    if gitignore_path.exists():
        content = gitignore_path.read_text()
        lines = content.splitlines()
        if ".env" not in lines and ".env\n" not in content:
            with open(gitignore_path, "a") as f:
                if content and not content.endswith("\n"):
                    f.write("\n")
                f.write(gitignore_entry)
            print("  Added .env to .gitignore.")
        else:
            print("  .gitignore already ignores .env.")
    else:
        gitignore_path.write_text(gitignore_entry)
        print("  Created .gitignore with .env entry.")

    # ── Next steps ───────────────────────────────────────────────────────────
    print()
    print("Fill in the missing values in .env then run:")
    print("  source .env && python training/modal_train.py --local --stage static_only --steps 5000")
    print()


if __name__ == "__main__":
    main()
