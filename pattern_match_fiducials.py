import logging
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imageFilepath', help="The filepath to the input image. Default: './images/BBB.png'", default='./images/BBB.png')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("pattern_match_fiducials.py main()")

    # Load the input image
    original_img = cv2.imread(args.imageFilepath)



if __name__ == '__main__':
    main()