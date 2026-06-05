import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs


class TrainArgs(BaseArgs):
    config_path: str
    secrets_path: str
    skip_cache_validation: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def parse_args() -> "TrainArgs":
        parser = argparse.ArgumentParser(description="One Trainer Training Script.")

        # @formatter:off

        parser.add_argument(
            "--config-path", type=str, required=True, dest="config_path", help="The path to the config file"
        )
        parser.add_argument(
            "--secrets-path", type=str, required=False, dest="secrets_path", help="The path to the secrets file"
        )
        parser.add_argument(
            "--callback-path",
            type=str,
            required=False,
            dest="callback_path",
            help="The path to the callback pickle file",
        )
        parser.add_argument(
            "--command-path", type=str, required=False, dest="command_path", help="The path to the command pickle file"
        )
        parser.add_argument(
            "--skip-cache-validation",
            action="store_true",
            dest="skip_cache_validation",
            help="Trust the existing cache index without re-validating mtimes/hashes. Files missing from the index are still cached. Useful on cloud storage where per-file stat calls are expensive.",
        )

        # @formatter:on

        args = TrainArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> "TrainArgs":
        data = []

        # name, default value, data type, nullable
        data.append(("config_path", None, str, True))
        data.append(("secrets_path", None, str, True))
        data.append(("callback_path", None, str, True))
        data.append(("command_path", None, str, True))
        data.append(("skip_cache_validation", False, bool, False))

        return TrainArgs(data)
