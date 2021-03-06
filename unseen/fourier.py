# pylint: disable=E0401, E1101, C0103, C0411

"""
chapter 3 of computers...
"""

import time
import numpy as np
from matplotlib import pyplot as plt



def radius_of_gyration(points):
    """
    compute the radius of gyration of a point cloud
    """
    mean_point = points.mean(0)
    displacement_vectors = points - mean_point
    distances = np.sum(displacement_vectors**2)
    return np.sqrt(distances / points.shape[0])

def plot_radius_of_gyration():
    """
    quick demo for radius of gyration
    """

    corner = np.random.rand(2) * 3

    points = np.random.rand(400, 2) + corner
    rog = radius_of_gyration(points)
    centroid = points.mean(0)

    # make the figure with a function that is an acronym for something obvious
    fig = plt.gcf()
    # plot the points
    plt.scatter(points[:, 0], points[:, 1])
    # make a circle
    circle = plt.Circle(centroid, rog, color='r', fill=False)
    # do something that makes sense to someone who is not me in order to plot the circle
    fig.gca().add_artist(circle)
    # set the limits
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    # sacrifice a black cat to ensure a good harvest
    plt.axes().set_aspect('equal')
    # matplotlib is an absolute nightmare
    plt.show()




def power_spectrum(waveform, time_interval=0.001, algorithm="first"):
    """
    not-optimized power spectrum decomposition. the interesting part (to me) is translating 
    pickover's pseudocode into something python-like.
    waveform should be a 1d array.

    pickover's indices (i and f) start at 1. i'm starting at 0. that changes the outcome of the
    algorithm, but not by much.
    """

    max_frequency = int(np.floor(1 / (2 * time_interval)))
    num_points = waveform.size

    if algorithm == "naive":
        # let's translate pseudocode directly
        power = np.empty(max_frequency, dtype=np.float64)
        for f in range(max_frequency):
            real = 0
            imaginary = 0
            arg = 2 * np.pi * f * time_interval
            for i in range(num_points):
                real += waveform[i] * np.cos(arg * i)
                imaginary += waveform[i] * np.sin(arg * i)
            power[f-1] = real **2 + imaginary ** 2

    elif algorithm == "first":
        # first try at vectorizing it for numpy. results are a little different (floating point
        # addition is not commutative) but much faster. and results are closer to numpy 
        # fast fourier transform.
        args_f = np.arange(max_frequency) * np.pi * 2 * time_interval
        args_if = np.stack([args_f * i for i in range(num_points)])
        real = (waveform.reshape(-1, 1) * np.cos(args_if)).sum(0)
        imaginary = (waveform.reshape(-1, 1) * np.sin(args_if)).sum(0)
        power = real**2 + imaginary**2

    elif algorithm == "fft":
        # results are different but much much faster. ignores our assertions of sampling rate obv.
        power = np.fft.fft(waveform)
        if power.dtype == np.complex128:
            power = np.real(power)**2 + np.imag(power)**2


    return power


def plot_power_spectrum(iterations=1):
    """
    get some random data and plot its power spectrum
    """

    algorithms = ["naive", "first", "fft"]
    data = np.concatenate([np.arange(10)+n for n in range(20)])

    for algorithm in algorithms:

        print("trying algorithm {}".format(algorithm))

        start = time.clock()
        for _ in range(iterations):        
            power = power_spectrum(data/10, algorithm=algorithm)
        

        finish = time.clock()

        print("{} iterations took {}s".format(iterations, finish - start))
        # print(power.shape)
        # print(power.max())
        # print(power.min())
        # print(power.mean())
        plt.figure()
        plt.plot(power)
        plt.show()

def compare():
    """
    compare the results of naive and first implementations of fourier transform
    """
    data = np.concatenate([np.arange(10)+n for n in range(20)])
    first = power_spectrum(data, algorithm="first")
    naive = power_spectrum(data, algorithm="naive")
    diff = naive - first
    print(diff.min())
    print(diff.max())
    print(diff.mean())
    plt.figure()
    plt.plot(diff)
    plt.show()


if __name__ == '__main__':
    # plot_power_spectrum()
    # compare()
    plot_radius_of_gyration()

