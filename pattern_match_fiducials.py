import logging
import cv2
import argparse
import os
import numpy as np
import utilities.blob_analysis as blob_analysis
import utilities.transformation as transformation
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--imageFilepath', help="The filepath to the input image. Default: './images/BBB.png'", default='./images/BBB.png')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--displayIntermediaryImages', help="Pause execution to display the intermediary images", action='store_true')
parser.add_argument('--fiducialInnerDiameterInPixels', help="The diameter of the fiducials' inner bright disk. Default: 26", type=int, default=26)
parser.add_argument('--fiducialOuterDiameterInPixels', help="The diameter of the fiducials' outer dark disk. Default: 68", type=int, default=68)
parser.add_argument('--numberOfFiducials', help="The number of fiducials. Default: 3", type=int, default=3)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')


def main():
    logging.info("pattern_match_fiducials.py main()")

    # Load the input image
    original_img = cv2.imread(args.imageFilepath)
    img_shapeHWC = original_img.shape

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Convert to grayscale
    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Create a synthetic fiducial image
    pattern_sizeHW = [args.fiducialOuterDiameterInPixels, args.fiducialOuterDiameterInPixels]
    if args.fiducialOuterDiameterInPixels %2 == 0:  # Make sure the pattern size is odd
        pattern_sizeHW[0] += 1
        pattern_sizeHW[1] += 1
    fiducial_pattern = np.zeros(pattern_sizeHW, dtype=np.uint8)
    cv2.circle(fiducial_pattern, (pattern_sizeHW[1]//2, pattern_sizeHW[0]//2),
               args.fiducialOuterDiameterInPixels//2, 70, cv2.FILLED)  # The outer disk is dark gray
    cv2.circle(fiducial_pattern, (pattern_sizeHW[1]//2, pattern_sizeHW[0]//2),
               args.fiducialInnerDiameterInPixels//2, 255, cv2.FILLED)  # The inner disk is white
    # Standardize the pattern image
    standardized_fiducial_pattern = (fiducial_pattern.astype(np.float32) - fiducial_pattern.mean())/fiducial_pattern.std()

    # Pattern match
    match_img = cv2.matchTemplate(grayscale_img.astype(np.float32), standardized_fiducial_pattern, cv2.TM_CCOEFF_NORMED)
    # Create an 8-bit version of the match image for visualization, padded with zeros to get an image the same size as the original
    padded_match_8bits_img = np.zeros((img_shapeHWC[0], img_shapeHWC[1]), dtype=np.uint8)
    padded_match_8bits_img[fiducial_pattern.shape[0]//2: fiducial_pattern.shape[0]//2 + match_img.shape[0],
        fiducial_pattern.shape[1]//2: fiducial_pattern.shape[1]//2 + match_img.shape[1]] = (128 * (match_img + 1.0)).astype(np.uint8)
    if args.displayIntermediaryImages:
        cv2.namedWindow("Padded match image")
        cv2.imshow("Padded match image", padded_match_8bits_img)
        cv2.waitKey(0)

    # Find the optimal threshold to detect the expected number of fiducials
    blob_detector = blob_analysis.BinaryBlobDetector()
    optimal_threshold = None
    optimal_seedPoint_boundingBox_list = None
    optimal_annotated_blobs_img = None
    for threshold in range(255, 1, -1):
        _, thresholded_img = cv2.threshold(padded_match_8bits_img, threshold, 255, cv2.THRESH_BINARY)
        # Count the number of blobs
        seedPoint_boundingBox_list, annotated_blobs_img = blob_detector.DetectBlobs(thresholded_img)
        logging.info("threshold = {}; len(seedPoint_boundingBox_list) = {}".format(threshold, len(seedPoint_boundingBox_list) ))
        if len(seedPoint_boundingBox_list) >= args.numberOfFiducials:
            optimal_threshold = threshold
            optimal_seedPoint_boundingBox_list = seedPoint_boundingBox_list
            optimal_annotated_blobs_img = annotated_blobs_img
            break
    logging.info("The optimal match threshold is {}. The number of found blobs is {}".format(optimal_threshold, len(optimal_seedPoint_boundingBox_list)))

    # Annotate the found fiducials
    annotated_img = copy.deepcopy(original_img)
    quadrant_to_pixels_location_dict = {}
    for (seed_point, bounding_box) in optimal_seedPoint_boundingBox_list:
        blob_center = (bounding_box[0] + bounding_box[2]//2, bounding_box[1] + bounding_box[3]//2)
        if blob_center[0] < img_shapeHWC[1]//2 and blob_center[1] < img_shapeHWC[0]//2:
            quadrant_to_pixels_location_dict['NorthWest'] = blob_center
        elif blob_center[0] > img_shapeHWC[1]//2 and blob_center[1] < img_shapeHWC[0]//2:
            quadrant_to_pixels_location_dict['NorthEast'] = blob_center
        elif blob_center[0] < img_shapeHWC[1]//2 and blob_center[1] > img_shapeHWC[1]//2:
            quadrant_to_pixels_location_dict['SouthWest'] = blob_center
        else:
            raise ValueError("pattern_match_fiducials.py main(): The blob center {} is not in one of the expected quadrants".format(blob_center))
        cv2.rectangle(annotated_img, (blob_center[0] - args.fiducialOuterDiameterInPixels//2, blob_center[1] - args.fiducialOuterDiameterInPixels//2),
                      (blob_center[0] + args.fiducialOuterDiameterInPixels // 2, blob_center[1] + args.fiducialOuterDiameterInPixels // 2),
                      color=(0, 255, 0), thickness=3)
        cv2.circle(annotated_img, blob_center, args.fiducialInnerDiameterInPixels//2 + 2, (0, 0, 255))

    # Compute the homography between coordinates in mm and coordinates in pixels
    # Note: The fiducial locations were measured manually with a caliper: they are certainly not exact
    position_mm_to_pixels_dict = {}
    position_mm_to_pixels_dict[(-38.33, 24.50)] = quadrant_to_pixels_location_dict['NorthWest']
    position_mm_to_pixels_dict[(29.93, 19.90)] = quadrant_to_pixels_location_dict['NorthEast']
    position_mm_to_pixels_dict[(-25.97, -19.90)] = quadrant_to_pixels_location_dict['SouthWest']
    mm_to_pixels_transformation_mtx = transformation.MillimetersToPixelsTransformationMatrix(position_mm_to_pixels_dict)

    # Compute the estimated location of the corners
    corner_to_pixels = {}
    corner_to_pixels['NorthWest'] = transformation.Transform(mm_to_pixels_transformation_mtx,
                                                             (-43.33, 27.33))
    corner_to_pixels['NorthEast'] = transformation.Transform(mm_to_pixels_transformation_mtx,
                                                             (43.33, 27.33))
    corner_to_pixels['SouthEast'] = transformation.Transform(mm_to_pixels_transformation_mtx,
                                                             (43.33, -27.33))
    corner_to_pixels['SouthWest'] = transformation.Transform(mm_to_pixels_transformation_mtx,
                                                             (-43.33, -27.33))
    # Draw the outline of the PCB
    cv2.line(annotated_img, corner_to_pixels['NorthWest'], corner_to_pixels['NorthEast'], (255, 0, 0), thickness=3)
    cv2.line(annotated_img, corner_to_pixels['NorthEast'], corner_to_pixels['SouthEast'], (255, 0, 0), thickness=3)
    cv2.line(annotated_img, corner_to_pixels['SouthEast'], corner_to_pixels['SouthWest'], (255, 0, 0), thickness=3)
    cv2.line(annotated_img, corner_to_pixels['SouthWest'], corner_to_pixels['NorthWest'], (255, 0, 0), thickness=3)
    if args.displayIntermediaryImages:
        cv2.namedWindow("Annotated image")
        cv2.imshow("Annotated image",annotated_img)
        cv2.waitKey(0)

    # Save intermediary images
    cv2.imwrite(os.path.join(args.outputDirectory, "grayscale.png"), grayscale_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "fiducialPattern.png"), fiducial_pattern)
    cv2.imwrite(os.path.join(args.outputDirectory, "match.png"), padded_match_8bits_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "blobs.png"), optimal_annotated_blobs_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "foundFiducials.png"), annotated_img)


if __name__ == '__main__':
    main()