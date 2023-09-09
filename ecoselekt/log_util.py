import logging

from ecoselekt.settings import settings


def get_logger(name="ecoselekt.log", exp_id=settings.EXP_ID) -> logging.Logger:
    logger = logging.getLogger(__name__)

    if logger.hasHandlers():
        # Logger is already configured, remove all handlers
        logger.handlers = []

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(name)

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - {exp_id} - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    c_handler.setLevel(logging.DEBUG)
    c_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)

    return logger
