from common import *
from pathlib import Path
from typing import *
from utils.misc_utils import load_json_with_vars


class RuntimeContext:
    """ A class holding the directories to use for model runtimes (train/inference). """

    runtime_dir: str
    config_file: str
    config: Dict[str, Any]
    model_name: str
    checkpoints_dir: str
    latest_checkpoint_file: str
    runs_dir: str

    def __init__(self, runtime_dir: str, config_file: str):
        self.runtime_dir = runtime_dir
        if not path.isdir(self.runtime_dir):
            raise Exception(f"Invalid runtime directory: {self.runtime_dir}")
        self.config_file = config_file
        if not path.isfile(self.config_file):
            raise Exception(f"Invalid config file: {config_file}")
        self.config = load_json_with_vars(self.config_file)

        self.model_name = Path(config_file).stem

        self.checkpoints_dir = path.join(runtime_dir, f'checkpoints-{self.model_name}')
        self.latest_checkpoint_file = path.join(self.checkpoints_dir, 'checkpoint-latest.pt')
        self.runs_dir = path.join(runtime_dir, f'runs-{self.model_name}')

        self._setup()


    def _setup(self):
        Path(self.runtime_dir).mkdir(exist_ok=True)
        Path(self.checkpoints_dir).mkdir(exist_ok=True)
        Path(self.runs_dir).mkdir(exist_ok=True)

    def describe(self):
        logging.info(f"[RuntimeContext] Information: ")
        logging.info(f"  Runtime directory: {self.runtime_dir}")
        logging.info(f"  Config file: {self.config_file}")
        logging.info(f"  Model name: {self.config_file}")
        logging.debug(f"  Checkpoints directory: {self.model_name}")
        logging.debug(f"  Runs directory: {self.checkpoints_dir}")


def parse_runtime_context_from_cmdline() -> RuntimeContext:
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)

    args = parser.parse_args()
    return RuntimeContext(args.runtime_dir, args.config_file)
