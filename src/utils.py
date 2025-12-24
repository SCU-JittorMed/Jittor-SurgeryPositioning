import time
import logging

def setup_logging(config):
    logging.basicConfig(
        level=getattr(logging, config['logging']['level'], logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )

def retry_on_error(max_retries=3, retry_delay=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
            return None
        return wrapper
    return decorator
