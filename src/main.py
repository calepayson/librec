import logging

from download import download
from exploration import explore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Getting raw data...")
    download()

    logger.info("Starting basic data exploration...")
    explore()


if __name__ == "__main__":
    main()
