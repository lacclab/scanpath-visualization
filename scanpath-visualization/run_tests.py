#!/usr/bin/env python3
"""Simple test runner script."""

import sys
import subprocess

def main():
    """Run pytest tests."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=".",
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest pytest-cov")
        sys.exit(1)

if __name__ == "__main__":
    main()
