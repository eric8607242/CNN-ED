import args
import yaml
from pprint import pprint

from agent import Trainer


def main(config_path):
    with open(config_path) as f:
        config = yaml.full_load(f)
        pprint(config)

    agent = Trainer(config)

    if config["train"]["resume"]:
        agent.resume()

    agent.train()

if __name__ == "__main__":
    parser = argparser.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to configuration file")
    args = vars(parser.parse_args())
    main(args["config"])
