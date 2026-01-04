import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_name=None):
    """
    Sets up a simple logger that writes to a file and stdout.

    Parameters
    ----------
    log_dir : str
        Directory to save log files.
    log_name : str or None
        Name of the log file (without extension). If None, uses timestamp.

    Returns
    -------
    logger : logging.Logger
        Configured logger object.
    """

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Default log filename with timestamp
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"log_{timestamp}.log"

    log_path = os.path.join(log_dir, log_name)

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized. Writing to {log_path}")

    return logger
