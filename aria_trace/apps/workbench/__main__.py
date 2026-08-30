"""Launch the canonical AriaTrace Workbench application."""

import multiprocessing

from .application import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
