#!/usr/bin/env bash
set -euo pipefail

# Minimal setup script
# - checks python available
# - creates a virtualenv directory named "uv" if missing
# - activates it (prints activation commands for the current shell)
# - installs requirements.txt using the venv pip

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/uv"
REQ_FILE="$REPO_DIR/requirements.txt"

info() { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
err() { printf "[ERROR] %s\n" "$*"; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

info "Running setup in: $REPO_DIR"

# Find a python executable
PYTHON=""
for p in python3 python; do
	if have_cmd "$p"; then
		PYTHON="$p"
		break
	fi
done

if [ -z "$PYTHON" ]; then
	err "No python interpreter found in PATH. Please install Python 3 and re-run this script."
fi

info "Using python: $(command -v "$PYTHON")"

# Ensure venv module is available
if "$PYTHON" -c "import venv" >/dev/null 2>&1; then
	info "Python venv module is available."
else
	warn "Python venv module not available. Trying to install virtualenv via pip."
	if have_cmd pip3; then
		PIP_CMD=pip3
	elif have_cmd pip; then
		PIP_CMD=pip
	else
		err "pip not found. Please install pip for your Python installation."
	fi
	info "Installing virtualenv with $PIP_CMD"
	"$PIP_CMD" install --user virtualenv
	# Re-check
	if ! "$PYTHON" -c "import venv" >/dev/null 2>&1; then
		warn "venv still not available; will use virtualenv package from user site-packages."
	fi
fi

# Create the virtual environment if it doesn't exist
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ] || [ -f "$VENV_DIR/Scripts/activate" ]; then
	info "Virtual environment 'uv' already exists at $VENV_DIR"
else
	info "Creating virtual environment at $VENV_DIR"
	# Prefer builtin venv
	if "$PYTHON" -c "import venv" >/dev/null 2>&1; then
		"$PYTHON" -m venv "$VENV_DIR"
	else
		# fallback to virtualenv
		if have_cmd virtualenv; then
			virtualenv -p "$PYTHON" "$VENV_DIR"
		else
			# try using python -m virtualenv (may be installed in user site)
			"$PYTHON" -m virtualenv -p "$PYTHON" "$VENV_DIR"
		fi
	fi
	info "Created virtualenv."
fi

# Activate venv for current shell if running under bash/zsh
ACTIVATED=false
if [ -n "${BASH_VERSION-}" ] || [ -n "${ZSH_VERSION-}" ]; then
	if [ -f "$VENV_DIR/bin/activate" ]; then
		# shellcheck disable=SC1090
		. "$VENV_DIR/bin/activate"
		ACTIVATED=true
		info "Activated virtualenv in current shell."
	fi
fi

# On Windows (PowerShell/CMD) we can't reliably activate from a bash script.
if ! $ACTIVATED; then
	if [ -f "$VENV_DIR/Scripts/activate" ]; then
		warn "Virtualenv exists but wasn't activated in this shell."
		printf "To activate in PowerShell: \n  & '%s\\Scripts\\Activate.ps1'\n" "$VENV_DIR"
		printf "To activate in cmd.exe: \n  %s\\Scripts\\activate.bat\n" "$VENV_DIR"
		printf "To activate in bash (WSL/Git-Bash):\n  source '%s/bin/activate'\n" "$VENV_DIR"
	fi
fi

# Install requirements if file exists
if [ -f "$REQ_FILE" ]; then
	info "Installing requirements from $REQ_FILE"
	# Ensure pip exists in the venv
	if $ACTIVATED; then
		PIP_EXE=pip
	else
		# prefer venv pip path
		if [ -f "$VENV_DIR/bin/pip" ]; then
			PIP_EXE="$VENV_DIR/bin/pip"
		elif [ -f "$VENV_DIR/Scripts/pip.exe" ]; then
			PIP_EXE="$VENV_DIR/Scripts/pip.exe"
		else
			err "pip not found inside virtualenv at $VENV_DIR"
		fi
	fi

	"$PIP_EXE" install -r "$REQ_FILE"
	info "Installed requirements."
else
	warn "No requirements.txt found at $REQ_FILE — skipping pip install step."
fi

info "Done."
