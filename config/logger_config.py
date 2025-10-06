import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import time
from pathlib import Path


class Logger:
    def __init__(
        self,
        log_dir="logs",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        app_name="app",
    ):
        Path(log_dir).mkdir(exist_ok=True)

        self.logger = logging.getLogger(app_name)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.logger.setLevel(logging.DEBUG)

        # Console handler
        self._add_handler(
            logging.StreamHandler(),
            console_level,
            logging.Formatter("%(message)s")
        )

        # Detailed log file handler
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(funcName)s - %(message)s"
        )
        self._add_handler(
            RotatingFileHandler(
                f"{log_dir}/detailed_logs.log",
                maxBytes=10*1024*1024,
                backupCount=2
            ),
            file_level,
            detailed_formatter
        )

        # Daily processing log handler
        proc_handler = TimedRotatingFileHandler(
            f"{log_dir}/processing.log",
            when="midnight",
            backupCount=7,
            encoding="utf-8"
        )
        proc_handler.suffix = "%Y-%m-%d"
        self._add_handler(
            proc_handler,
            logging.INFO,
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Error log handler
        error_formatter = logging.Formatter(
            "%(asctime)s - %(process)d - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
        self._add_handler(
            RotatingFileHandler(
                f"{log_dir}/errors.log",
                maxBytes=5*1024*1024,
                backupCount=1
            ),
            logging.ERROR,
            error_formatter
        )

        self.logger.info(
            "Logger initialized at %s",
            time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _add_handler(self, handler, level, formatter):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


logger = Logger().logger
