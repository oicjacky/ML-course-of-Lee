import configparser
import time
import logging

def setup_logger(name, log_file =None, level=logging.INFO):
    """ Setup different loggers to record the message with corresponding `level` and `file name` """
    if 'info' in name:
        formatter = logging.Formatter('[%(asctime)s]: %(message)s')
    elif 'error' in name:
        formatter = logging.Formatter('[%(asctime)s] - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    else:
        formatter = logging.Formatter('[%(asctime)s] - %(pathname)s[line:%(lineno)d]: %(message)s')
    if log_file:
        handler = logging.FileHandler(log_file, encoding= 'utf-8')
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

config = configparser.ConfigParser()
config.read(r'./config.ini')
log_file = config.get('File', 'log_file')
LOGGER = setup_logger(name="datetime_logger", log_file=f"{log_file+time.strftime('_%H%M')}.log")

if __name__ == "__main__":
    
    logger = setup_logger(name="main_logger", log_file="example.log")
    logger.info(f"Hello, i am {logger.name}!")