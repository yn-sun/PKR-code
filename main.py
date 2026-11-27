import argparse
import json

from trainer import train


def main():
    parser = argparse.ArgumentParser(description="PKR: class-incremental learning.")
    parser.add_argument("--config", type=str, default="./exps/pkr_cifar100_b10_inc10.json",
                        help="Json file of settings.")
    args = vars(parser.parse_args())
    with open(args["config"]) as f:
        args.update(json.load(f))
    train(args)


if __name__ == "__main__":
    main()
