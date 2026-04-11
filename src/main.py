import argparse
import logging

from download import download
from exploration import explore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all resources")
    args = parser.parse_args()

    if args.rebuild:
        logger.info("Getting raw data...")
        download(rebuild=True)

    logger.info("Starting basic data exploration...")
    explore()


if __name__ == "__main__":
    main()
