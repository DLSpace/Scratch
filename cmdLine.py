import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", help="load saved model",
                    action="store_true")

parser.add_argument("-p", "--patience", help="patience level")

args = parser.parse_args()

print(args)
print(args.patience.__class__)
