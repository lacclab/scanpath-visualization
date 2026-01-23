import sys
import importlib.resources as resources
from typing import Optional

from streamlit.web import cli as stcli


def main(argv: Optional[list[str]] = None) -> None:
    """Entrypoint that launches the Streamlit app via ``streamlit run``."""
    extra_args = list(argv) if argv is not None else sys.argv[1:]
    app_resource = resources.files(__package__).joinpath("app.py")
    with resources.as_file(app_resource) as app_path:
        sys.argv = ["streamlit", "run", str(app_path), *extra_args]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
