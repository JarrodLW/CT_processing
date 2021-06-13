#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:56:24 2020

@author: jlw31
"""

import numpy as np
import itertools
import scipy.stats as ss
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


def circle_mask(width, ratio):
    # taken from Vo's code
    """
    Create a circle mask.
    ---------
    Parameters: - width: width of a square array.
                - ratio: ratio between the diameter of the mask and
                    the width of the array.
    ---------
    Return:     - square array.
    """
    mask = np.zeros((width, width), dtype=np.float32)
    center = width // 2
    radius = ratio * center
    y, x = np.ogrid[-center:width - center, -center:width - center]
    mask_check = x * x + y * y <= radius * radius
    mask[mask_check] = 1.0
    return mask


def pad_sino(sino_array, a_step, a_min, num_angles, a_list):
    """
     takes a sino and inserts zeros for the missing angle entries
     takes a list of floats, representing the angles at which measurements were taken, along with a_min, a_max
     defining range of available angles and the angle step
     returns: a padded sinogram array, along with the corresponding mask to be used in reconstructions
    """

    a_max = a_min + (num_angles - 1) * a_step

    assert len(a_list) == sino_array.shape[0], "number of measurements not consistent with sinogram dims"
    assert a_min <= min(a_list), "minimum angle too large"
    assert a_max >= max(a_list), "maximum angle too small"

    ind_known_angles = np.array((np.array(a_list) - a_min) / a_step, dtype=int)
    mask = np.zeros((num_angles, sino_array.shape[1]))
    mask[ind_known_angles, :] = 1

    new_sino_array = np.zeros((num_angles, sino_array.shape[1]))

    for ind, ind_angle in enumerate(ind_known_angles):
        new_sino_array[ind_angle, :] = sino_array[ind, :]

    return new_sino_array, mask


def equipartition(num_bins, num_examples):
    # splits a given number of bins into roughly equal blocks, and samples one example from each

    rem = num_bins % num_examples
    div = num_bins // num_examples
    interval_lengths = np.array([div] * (num_examples - rem) + [div + 1] * rem)
    interval_lengths = list(np.random.permutation(interval_lengths))
    interval_positions = list(itertools.accumulate(interval_lengths))
    intervals = np.split(np.arange(num_bins), interval_positions)
    #bin_indices = np.sort(np.asarray([np.random.choice(intervals[i]) for i in range(num_examples)]))
    bin_indices = [intervals[i][len(intervals[i])//2] for i in range(num_examples)]

    return bin_indices


def horiz_rand_walk_mask(height, width, num_walks, distr='equipartition', allowing_inter=True, p=[0.25, 0.5, 0.25], scale=None):
    """
     constructs random walks left to right across a rectangular grid
     p gives the probabilities of moving down a pixel, staying put and moving up a pixel
    """

    if distr == 'equipartition':
        # splits the row space into roughly equal intervals, and samples one starting position from each
        initial_position = equipartition(height, num_walks)

    if distr == 'centre_clustered':

        # TO DO: ensure that the correct number of points are being sampled: right now I'm sampling with replacement!

        # use equipartition and apply the inverse of a multinomial distribution which approximates the normal
        # centred at the midpoint of the data with give variance
        x = np.arange(-height // 2, height // 2)
        #xU, xL = x + 0.5, x - 0.5
        #prob = ss.norm.cdf(xU, scale=50) - ss.norm.cdf(xL, scale=50)
        #prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        cumul_probs = np.append([0], ss.norm.cdf(x, scale=scale))
        sample_locations = equipartition(height, num_walks)/height
        hist = np.histogram(sample_locations, cumul_probs)[0]
        hist_top_half = hist[0:height//2]
        hist_bottom_half = hist[height//2:]

        for i in range(1, len(hist_top_half)):
            if hist_top_half[-i] > 1:
                hist_top_half[-i - 1] += hist_top_half[-i] - 1
                hist_top_half[-i] = 1

        for i in range(len(hist_bottom_half)-1):
            if hist_bottom_half[i] > 1:
                hist_bottom_half[i + 1] += hist_bottom_half[i] - 1
                hist_bottom_half[i] = 1

        hist_modified = np.concatenate([hist_top_half, hist_bottom_half])
        initial_position = np.nonzero(hist_modified)[0]

        #initial_position = np.nonzero(np.histogram(sample_locations, cumul_probs)[0])[0]

    elif distr == 'uniform':
        initial_position = np.sort(np.random.choice(range(height), replace=False, size=num_walks))

    # elif distr == 'multinomial':
    #     x = np.arange(-height // 2, height // 2)
    #     xU, xL = x + 0.5, x - 0.5
    #     prob = ss.norm.cdf(xU, scale=50) - ss.norm.cdf(xL, scale=50)
    #     prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    #
    #     equipartition(num_bins, num_examples)
    #
    #     nums = np.random.choice(x, size=num_walks, replace=False, p=prob)
    #     initial_position = nums + height // 2

    # else:
    #     print('Initial distribution type not supported')
    #     raise

    position = initial_position

    walk_array = np.zeros((height, width))
    walk_array[initial_position, 0] = 1

    for i in range(1, width):
        shift = np.random.choice([-1, 0, 1], size=num_walks, p=p)

        if allowing_inter:
            position += shift

        else:
            position_roll_up = np.roll(position, -1)
            position_roll_down = np.roll(position, 1)
            position_roll_up[-1] = height
            position_roll_down[0] = 0
            position += shift
            position = np.maximum(position, position_roll_down + 1)
            position = np.minimum(position, position_roll_up - 1)

        position = np.minimum(np.maximum(position, 0), height - 1)
        walk_array[position, i] = 1

    transparency = np.count_nonzero(walk_array) / (height * width)

    return walk_array, transparency


def vert_rand_walk_mask(height, width, num_walks, distr='equipartition', allowing_inter=True, p=[0.25, 0.5, 0.25]):
    """performs vertical random walks over a grid, top to bottom."""

    walk_array, sparsity = horiz_rand_walk_mask(width, height, num_walks, distr=distr,
                                                allowing_inter=allowing_inter, p=p)
    walk_array = walk_array.T

    return walk_array, sparsity


def bernoulli_mask(height, width, expected_sparsity=0.5):
    mask = np.random.binomial(1, expected_sparsity, size=(height, width))
    sparsity = np.count_nonzero(mask) / np.prod(np.shape(mask))

    return mask, sparsity


# def radial_spoke_mask(height, width, num_spokes):
#
#     slopes = np.arange(num_spokes)*2*np.pi/num_spokes
#     mask = np.zeros((height, width))
#     #centre = (height//2, width//2)
#
#     for slope in slopes:
#         heights = height//2 + np.tan(slope)*(np.arange(width) - width//2)
#         np.histogram(heights, )


def sequence_partition(len_of_sequence, initial_division):
    """takes the sequence [0, 1, ..., N-1] and successively partitions it by breaking up the largest existing
     block into two (roughly equal-sized) blocks
     returns the points at which successive divisions take place
    """

    division_list = [initial_division]
    sequence = list(np.arange(len_of_sequence))

    if initial_division == 0:
        partition = [sequence]

    else:
        partition = [sequence[: initial_division],
                     sequence[initial_division:]]

    while len(partition) < len_of_sequence:
        length_list = [len(_) for _ in partition]
        division_index = np.argmax(np.array(length_list))
        length = np.amax(np.array(length_list))
        part = partition[division_index]
        part1 = part[0: length // 2]
        part2 = part[length // 2:]
        division_list.append(part2[0])
        partition.pop(division_index)
        partition.append(part1)
        partition.append(part2)

    if initial_division != 0:
        division_list.append(0)

    return division_list


def recasting_fourier_as_complex(vec, height, width):
    # Melanie's raw data files consist of a vector of real coefficients

    assert len(vec) == height * width, "prescribed dimensions inconsistent with length of array"

    fourier_coeff_real_im = np.reshape(vec, (height, width))
    fourier_coeff_real = fourier_coeff_real_im[:, ::2]
    fourier_coeff_im = fourier_coeff_real_im[:, 1::2]

    fourier_coeff = fourier_coeff_real + fourier_coeff_im * 1j

    return fourier_coeff

def recon_error(im_recon, im_ref):

    l2_error = np.sqrt(np.sum(np.square(im_recon - im_ref)))
    psnr_val = psnr(im_recon, im_ref, data_range=im_recon.max() - im_recon.min())
    ssim_val = ssim(im_ref, im_recon, data_range=im_recon.max() - im_recon.min())

    #print('l2_error:'+str(l2_error))

    return l2_error, psnr_val, ssim_val

def unpacking_fourier_coeffs_15032021(arr):

    fourier_real_im = arr[:, :64]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

# def create_synthetic_data(im_stack, measurement_type):
#     # takes a stack of images (given as numpy array) and produces synthetic data
#     # by applying the appropriate forward operator
#
#     height, width = im_stack.shape
#     image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
#                                         shape=[height, width], dtype='float')
#
#     if measurement_type = 'CT':
#
#         # Make a parallel beam geometry with flat detector
#         # Angles: uniformly spaced, n = 360, min = 0, max = pi
#         angle_partition = odl.uniform_partition(0, np.pi, data_stack.shape[1])
#         # Detector: uniformly sampled, n = 512, min = -30, max = 30
#         detector_partition = odl.uniform_partition(-30, 30, data_stack.shape[2])
#         geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
#
#         # Create the forward operator
#         forward_op = odl.tomo.RayTransform(image_space, geometry, impl='astra_cpu')
#
#     else:
#         raise ValueError("Measurement type "+str(measurement_type)+" not implemented")
#
#     synth_data_stack = np.zeros()
#
#     for i in range(im_stack.shape[0]):
#
#         forward_op(image_space.element(im-stack[i, :, :])).asarray()
