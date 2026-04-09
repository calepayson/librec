import logging

from download import download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Getting raw data...")
    download()


if __name__ == "__main__":
    main()
