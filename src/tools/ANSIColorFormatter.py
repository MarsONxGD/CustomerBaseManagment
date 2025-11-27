import logging


class ANSIColorFormatter(logging.Formatter):
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    CYAN = "\x1b[36;20m"
    GREY = "\x1b[38;20m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record):
        log_message = super().format(record)
        color = self.LEVEL_COLORS.get(record.levelno, self.GREY)
        return color + log_message + self.RESET
