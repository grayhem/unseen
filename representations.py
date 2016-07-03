# pylint: disable=E0401, E1101, C0103, C0411

"""
chapter 4 of computers...
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# import visual_fields


def test_sdp():
    """
    do a quick symmetrized dot pattern
    """

    waveform = np.concatenate([np.arange(100)+n for n in range(20)]).astype(np.float64)
    frame_number = 0
    while True:
        # use_waveform = waveform + np.random.rand(2000)*5
        image = (symmetrized_dots(
            waveform,
            top=300,
            lag=frame_number % 100)*255).astype(np.uint8)
        # print(image.dtype)
        cv2.imshow('frame', image)
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def symmetrized_dots(waveform, angle=60, top=50, lag=1):
    """
    draw a symmetrical snowflake-like image (boolean array) from a 1d waveform
    """

    dot_indices = symmetrized_dot_cloud(waveform, angle=angle, top=top, lag=lag)

    output = np.zeros((top*2, top*2), dtype=np.bool)

    # draw dots on the output image
    output_index = np.ravel_multi_index(dot_indices, output.shape, mode='clip')
    np.put(output, output_index, True)

    return output


def symmetrized_dot_cloud(waveform, angle=60, top=50, lag=1):
    """
    same as symmetrized dots but returns a 2d point cloud instead of a boolean image. note the
    return value is a tuple (x, y) of int ndarrays, which is equivalent to (cols, rows).
    """

    # normalize the waveform to 0-top 
    waveform = normalize(waveform, top)

    # roll the waveform by lag to get the angles we'll use
    roll_waveform = np.roll(waveform, lag)

    # now compute the angles
    first_angle = np.concatenate(
        [base_angle + roll_waveform for base_angle in np.arange(0, 360, angle)])
    second_angle = np.concatenate(
        [base_angle - roll_waveform for base_angle in np.arange(0, 360, angle)])
    all_angles = np.deg2rad(np.concatenate([first_angle, second_angle]))

    # tile the original waveform until it matches the length of the angles
    num_repeats = int(all_angles.size / waveform.size)
    waveform = np.tile(waveform, num_repeats)

    # now make the point cloud
    x_cols = waveform * np.cos(all_angles) + top
    y_rows = waveform * np.sin(all_angles) + top

    return (np.fix(x_cols).astype(np.int), np.fix(y_rows).astype(np.int))

def normalize(waveform, top):
    """
    normalize the range of waveform to 0-top
    """

    hi = waveform.max()
    lo = waveform.min()
    return (waveform - lo) * top / (hi - lo)


def test_normalize():
    """
    test the normalizing function
    """
    waveform = np.arange(100)
    top = 20
    waveform = normalize(waveform, top)
    assert waveform.min() == 0.0
    assert waveform.max() == 20.0


if __name__ == '__main__':
    test_sdp()

