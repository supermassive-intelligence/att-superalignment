from infra.salign.cli.solve import solve
from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    arguments = parse_arguments()

    if arguments.command == "solve":
        solve()
    elif arguments.command is None:
        print("No command specified.")


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("snowflake.connector.connection").setLevel(logging.WARNING)


def parse_arguments():
    parser = ArgumentParser(
        description="The command line interface for salign - the superalignment tool for large language models."
    )

    parser.add_argument('command', help='Superalignment command.')

    argumments = parser.parse_args()
    return argumments

main()
