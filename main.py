import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Script to generate Trump tweets')
    parser.add_argument('--data_path', '-d', metavar='STRING', default='./data',
                        help="Where the CSV with the tweets is stored")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='./output',
                        help="Where the output will be stored")

    return parser.parse_args()


class TextGeneration:
    def __init__(self):
        # Here we init stuff
        self.args = get_args()
        print("Init")

    def run(self):
        print("RENNEN MIENJONG!")


def main():
    generation = TextGeneration()
    generation.run()


if __name__ == '__main__':
    main()
