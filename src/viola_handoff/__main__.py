"""Allow ``python -m viola_handoff`` to invoke the CLI."""

from .cli import main

raise SystemExit(main())
