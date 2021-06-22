import logging
import sys


class Logger:
    """Initialize project-wide logger. The logger outputs to both stdout and a file."""

    # output format for log string
    LOG_FORMAT_STRING = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    def __init__(self):

        self.log_format = logging.Formatter(self.LOG_FORMAT_STRING)
        self.logger = self.set_logger()
        self.initialize_logger()

    def set_logger(self):
        """Initialize logger as level info."""

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        return logger

    def initialize_logger(self):
        """Construct console handler and initialize logger to stdout."""

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)

    @staticmethod
    def close_logger():
        """Shutdown logger."""

        # Remove logging handlers
        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.shutdown()
