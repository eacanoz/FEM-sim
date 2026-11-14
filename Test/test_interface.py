"""Interface of the model (phase E of docs/PLAN_RENDIMIENTO_GPU_ML.md): the session log (one file, created on the first message,
folder and console level configurable).

Run from the project root:  python Test/test_interface.py
"""

import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Source.logger_config import configure_logging, sim_logger
import Source.logger_config as logger_config


def test_log_file_is_created_at_the_first_message_in_the_chosen_folder():
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, 'sub', 'logs')
        logger = configure_logging(log_dir=folder, console_level=None)
        assert logger is sim_logger
        assert not os.path.exists(folder)                       # nothing written yet
        logger.info("first message")
        files = glob.glob(os.path.join(folder, 'sim_*.log'))
        assert len(files) == 1
        logger.info("second message")
        assert len(glob.glob(os.path.join(folder, 'sim_*.log'))) == 1
        logger.debug("detail only for the file")
        for handler in logger.handlers:
            handler.flush()
        text = open(files[0], encoding='utf-8').read()
        assert "first message" in text and "second message" in text and "detail only" in text
        configure_logging(log_dir=None, console_level=None)     # release the file before deleting the folder


def test_levels_and_no_duplicated_handlers():
    logger = configure_logging(log_dir=None, console_level='WARNING')
    logger = configure_logging(log_dir=None, console_level='WARNING')
    assert len(logger.handlers) == 1 and logger.handlers[0].level == 30
    logger = configure_logging(log_dir=None, console_level=None)
    assert logger.handlers == []
    configure_logging()                                          # back to the default


if __name__ == "__main__":
    tests = [test_log_file_is_created_at_the_first_message_in_the_chosen_folder, test_levels_and_no_duplicated_handlers]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS ", t.__name__)
        except Exception as e:
            failed += 1
            print("FAIL ", t.__name__, repr(e))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
