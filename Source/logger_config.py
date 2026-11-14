import logging
import os
from datetime import datetime

LOGGER_NAME = "FEM_Simulator"
_FORMAT = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def setup_logger(name=LOGGER_NAME, log_dir="Logs", console_level=logging.INFO):

    """Logger of the session: one file per session in log_dir, created when the first message is written (importing
    Source no longer creates an empty file), and a console handler at console_level (None: no console output)."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)       # the file keeps everything

    # No duplicated handlers if a Jupyter cell runs again
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    if log_dir is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"sim_{timestamp}.log")
        file_handler = _LazyDirFileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_FORMAT)
        logger.addHandler(file_handler)

    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(_FORMAT)
        logger.addHandler(console_handler)

    return logger


class _LazyDirFileHandler(logging.FileHandler):

    """File handler that creates its folder and file at the first record (delay=True only postpones the file)."""

    def __init__(self, path):
        super().__init__(path, delay=True)

    def _open(self):
        os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
        return super()._open()


def configure_logging(log_dir="Logs", console_level="INFO"):

    """Configures the logging of the session: the folder of the log file (None: no file) and the console level
    ('DEBUG', 'INFO', 'WARNING', ... or None for silence). Starts a new file."""

    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper())

    return setup_logger(log_dir=log_dir, console_level=console_level)


# Global instance to import elsewhere
sim_logger = setup_logger()
