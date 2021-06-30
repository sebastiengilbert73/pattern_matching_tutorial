import cv2
import numpy as np
import math


def Transform(transformation_mtx, xy, round_values=True):
    if transformation_mtx.shape != (3, 3):
        raise ValueError("transformation.Transform(): transformation_mtx.shape {} != (3, 3)".format(transformation_mtx.shape))
    if len(xy) != 2:
        raise ValueError("transformation.Transform(): len(xy) ({}) != 2".format(len(xy)))
    x_arr = np.ones((3), dtype=float)
    x_arr[0] = xy[0]
    x_arr[1] = xy[1]

    x_prime_arr = transformation_mtx @ x_arr
    if round_values:
        return (round(x_prime_arr[0]), round(x_prime_arr[1]))
    return (x_prime_arr[0], x_prime_arr[1])

def CornerNameToPosition(center, dimensions, theta):
    corner_to_position_dict = {}
    Lx = dimensions[0]
    Ly = dimensions[1]
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    corner_to_position_dict['NorthWest'] = (center[0] - Lx/2 * cosT - Ly/2 * sinT,
                                            center[1] - Lx/2 * sinT + Ly/2 * cosT)
    corner_to_position_dict['NorthEast'] = (center[0] + Lx/2 * cosT - Ly/2 * sinT,
                                            center[1] + Lx/2 * sinT + Ly/2 * cosT)
    corner_to_position_dict['SouthEast'] = (center[0] + Lx/2 * cosT + Ly/2 * sinT,
                                            center[1] + Lx/2 * sinT - Ly/2 * cosT)
    corner_to_position_dict['SouthWest'] = (center[0] - Lx/2 * cosT + Ly/2 * sinT,
                                            center[1] - Lx/2 * sinT - Ly/2 * cosT)
    return corner_to_position_dict

def MillimetersToPixelsTransformationMatrix(position_mm_to_pixels_dic):
    if len(position_mm_to_pixels_dic) < 3:
        raise ValueError("transformation.MillimetersToPixelsTransformationMatrix(): The number of matches ({}) is less than 3".format(len(position_mm_to_pixels_dic)))
    A = np.zeros((2 * len(position_mm_to_pixels_dic), 6))
    b = np.zeros((2 * len(position_mm_to_pixels_dic), 1))
    position_mm_list = list(position_mm_to_pixels_dic) # List of keys: position_mm
    for correspondenceNdx in range(len(position_mm_list)):
        x = position_mm_to_pixels_dic[position_mm_list[correspondenceNdx]][0]
        y = position_mm_to_pixels_dic[position_mm_list[correspondenceNdx]][1]
        X = position_mm_list[correspondenceNdx][0]
        Y = position_mm_list[correspondenceNdx][1]
        A[2 * correspondenceNdx, 0] = X
        A[2 * correspondenceNdx, 1] = Y
        A[2 * correspondenceNdx, 2] = 1.0
        A[2 * correspondenceNdx + 1, 3] = X
        A[2 * correspondenceNdx + 1, 4] = Y
        A[2 * correspondenceNdx + 1, 5] = 1.0
        b[2 * correspondenceNdx, 0] = float(x)
        b[2 * correspondenceNdx + 1, 0] = float(y)
    # Solve the system of linear equations with least-squares
    coefs, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    # coefs.shape = (6, 1)
    millimeters_to_pixels_transformation_Mtx = np.zeros((3, 3))
    millimeters_to_pixels_transformation_Mtx[0, 0] = coefs[0, 0]
    millimeters_to_pixels_transformation_Mtx[0, 1] = coefs[1, 0]
    millimeters_to_pixels_transformation_Mtx[0, 2] = coefs[2, 0]
    millimeters_to_pixels_transformation_Mtx[1, 0] = coefs[3, 0]
    millimeters_to_pixels_transformation_Mtx[1, 1] = coefs[4, 0]
    millimeters_to_pixels_transformation_Mtx[1, 2] = coefs[5, 0]
    millimeters_to_pixels_transformation_Mtx[2, 2] = 1.0
    return millimeters_to_pixels_transformation_Mtx