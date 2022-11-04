from src.Application import Application
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='grid-ping')
    parser.add_argument('--root', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.root is None:
        os.chdir("../")
    Application().run()


