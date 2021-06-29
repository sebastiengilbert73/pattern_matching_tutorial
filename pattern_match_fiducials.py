import logging
import cv2
import argparse
import os
import numpy as np
import utilities.blob_analysis as blob_analysis

parser = argparse.ArgumentParser()
parser.add_argument('--imageFilepath', help="The filepath to the input image. Default: './images/BBB.png'", default='./images/BBB.png')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--fiducialInnerDiameterInPixels', help="The diameter of the fiducials' inner bright disk. Default: 26", type=int, default=26)
parser.add_argument('--fiducialOuterDiameterInPixels', help="The diameter of the fiducials' outer dark disk. Default: 68", type=int, default=68)
parser.add_argument('--numberOfFiducials', help="The number of fiducials. Default: 3", type=int, default=3)
parser.add_argument('--matchThresholdUpperLimit', help="The match threshold upper limit. Default: 255", type=int, default=255)
parser.add_argument('--matchThresholdLowerLimit', help="The match threshold lower limit. Default: 180", type=int, default=180)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("pattern_match_fiducials.py main()")

    # Load the input image
    original_img = cv2.imread(args.imageFilepath)
    img_shapeHWC = original_img.shape

    # Convert to grayscale
    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Create a synthetic fiducial image
    fiducial_pattern = np.zeros((args.fiducialOuterDiameterInPixels, args.fiducialOuterDiameterInPixels), dtype=np.uint8)
    cv2.circle(fiducial_pattern, (args.fiducialOuterDiameterInPixels//2, args.fiducialOuterDiameterInPixels//2),
               args.fiducialOuterDiameterInPixels//2, 70, cv2.FILLED)  # The outer disk is dark gray
    cv2.circle(fiducial_pattern, (args.fiducialOuterDiameterInPixels//2, args.fiducialOuterDiameterInPixels//2),
               args.fiducialInnerDiameterInPixels//2, 255, cv2.FILLED)  # The inner disk is white
    # Normalize the pattern image
    normalized_fiducial_pattern = (fiducial_pattern.astype(np.float32) - fiducial_pattern.mean())/fiducial_pattern.std()

    # Pattern match
    match_img = cv2.matchTemplate(grayscale_img.astype(np.float32), normalized_fiducial_pattern, cv2.TM_CCOEFF_NORMED)
    # Create an 8-bit version of the match image for visualization, padded with zeros to get an image the same size as the original
    padded_match_8bits_img = np.zeros((img_shapeHWC[0], img_shapeHWC[1]), dtype=np.uint8)
    padded_match_8bits_img[fiducial_pattern.shape[0]//2: fiducial_pattern.shape[0]//2 + match_img.shape[0],
        fiducial_pattern.shape[1]//2: fiducial_pattern.shape[1]//2 + match_img.shape[1]] = (128 * (match_img + 1.0)).astype(np.uint8)

    # Find the optimal threshold to detect the expected number of fiducials
    blob_detector = blob_analysis.BinaryBlobDetector()
    optimal_threshold = None
    optimal_seedPoint_boundingBox_list = None
    optimal_annotated_blobs_img = None
    for threshold in range(255, 1, -1):
        _, thresholded_img = cv2.threshold(padded_match_8bits_img, threshold, 255, cv2.THRESH_BINARY)
        # Count the number of blobs
        seedPoint_boundingBox_list, annotated_img = blob_detector.DetectBlobs(thresholded_img)
        logging.info("threshold = {}; len(seedPoint_boundingBox_list) = {}".format(threshold, len(seedPoint_boundingBox_list) ))
        if len(seedPoint_boundingBox_list) >= args.numberOfFiducials:
            optimal_threshold = threshold
            optimal_seedPoint_boundingBox_list = seedPoint_boundingBox_list
            optimal_annotated_blobs_img = annotated_img
            break
    logging.info("The optimal match threshold is {}. The number of found blobs is {}".format(optimal_threshold, len(optimal_seedPoint_boundingBox_list)))

    # Save intermediary images
    cv2.imwrite(os.path.join(args.outputDirectory, "grayscale.png"), grayscale_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "fiducialPattern.png"), fiducial_pattern)
    cv2.imwrite(os.path.join(args.outputDirectory, "match.png"), padded_match_8bits_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "blobs.png"), optimal_annotated_blobs_img)



if __name__ == '__main__':
    main()