"""Streamlit Community Cloud entry point.

Cloud deployments by convention look for ``streamlit_app.py`` at the repo root.
This is a thin shim that just calls the packaged ``main()``.
"""

from scanpath_studio.app import main

main()
