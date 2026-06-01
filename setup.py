import subprocess
import sys
import os
import venv

# ─────────────────────────────────────────────
#  CONFIGURATION — change these if needed
# ─────────────────────────────────────────────
VENV_NAME = "ir_env"
APP_FILE = "app.py"
REQUIREMENTS = "requirements.txt"


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def banner(text):
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}")


def get_venv_python():
    """Return path to the Python executable inside the venv."""
    if sys.platform == "win32":
        return os.path.join(VENV_NAME, "Scripts", "python.exe")
    return os.path.join(VENV_NAME, "bin", "python")


def get_venv_streamlit():
    """Return path to the Streamlit executable inside the venv."""
    if sys.platform == "win32":
        return os.path.join(VENV_NAME, "Scripts", "streamlit.exe")
    return os.path.join(VENV_NAME, "bin", "streamlit")


def run(cmd, description=""):
    """Run a command and exit cleanly on failure."""
    if description:
        print(f"\n  ▶ {description}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n  ✖ Failed: {' '.join(str(c) for c in cmd)}")
        print("    Fix the error above and re-run setup.py")
        sys.exit(1)


# ─────────────────────────────────────────────
#  STEP 1 — Check Python version
# ─────────────────────────────────────────────
banner("Step 1 — Checking Python version")
major, minor = sys.version_info.major, sys.version_info.minor
print(f"  Python {major}.{minor} detected")
if major < 3 or minor < 10:
    print("  ✖ Python 3.10 or higher is required.")
    print("    Download from: https://www.python.org/downloads/")
    sys.exit(1)
print("  ✔ Python version OK")

# ─────────────────────────────────────────────
#  STEP 2 — Create virtual environment
# ─────────────────────────────────────────────
banner("Step 2 — Creating virtual environment")
if os.path.exists(VENV_NAME):
    print(f"  ℹ '{VENV_NAME}' already exists — skipping creation")
else:
    print(f"  Creating '{VENV_NAME}'...")
    venv.create(VENV_NAME, with_pip=True)
    print(f"  ✔ Virtual environment created")

venv_python = get_venv_python()
venv_streamlit = get_venv_streamlit()

if not os.path.exists(venv_python):
    print(f"  ✖ Could not find Python at: {venv_python}")
    sys.exit(1)

# ─────────────────────────────────────────────
#  STEP 3 — Upgrade pip inside venv
# ─────────────────────────────────────────────
banner("Step 3 — Upgrading pip")
run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip...")
print("  ✔ pip upgraded")

# ─────────────────────────────────────────────
#  STEP 4 — Install libraries from requirements.txt
# ─────────────────────────────────────────────
banner("Step 4 — Installing libraries")
if not os.path.exists(REQUIREMENTS):
    print(f"  ✖ '{REQUIREMENTS}' not found in this folder.")
    print("    Make sure requirements.txt is next to setup.py")
    sys.exit(1)

run(
    [venv_python, "-m", "pip", "install", "-r", REQUIREMENTS],
    f"Installing from {REQUIREMENTS}...",
)
print("  ✔ All libraries installed")

# ─────────────────────────────────────────────
#  STEP 5 — Download spaCy model
# ─────────────────────────────────────────────
banner("Step 5 — Downloading spaCy model")
run(
    [venv_python, "-m", "spacy", "download", "en_core_web_sm"],
    "Downloading en_core_web_sm...",
)
print("  ✔ spaCy model ready")

# ─────────────────────────────────────────────
#  STEP 6 — Download NLTK data
# ─────────────────────────────────────────────
banner("Step 6 — Downloading NLTK data")
nltk_packages = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "omw-1.4",
]
for pkg in nltk_packages:
    run(
        [venv_python, "-c", f"import nltk; nltk.download('{pkg}', quiet=False)"],
        f"Downloading nltk: {pkg}",
    )
print("  ✔ All NLTK data downloaded")

# ─────────────────────────────────────────────
#  STEP 7 — Launch Streamlit app
# ─────────────────────────────────────────────
banner("Step 7 — Launching Streamlit")
if not os.path.exists(APP_FILE):
    print(f"  ✖ '{APP_FILE}' not found.")
    print("    Create app.py first, then re-run: python setup.py")
    sys.exit(1)

print(f"\n  ✔ Setup complete! Launching {APP_FILE}...")
print("  ℹ  Press Ctrl+C in this terminal to stop the app\n")

# Launch streamlit — this blocks until the user presses Ctrl+C
subprocess.run([venv_streamlit, "run", APP_FILE])
