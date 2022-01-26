# ISTA 421 / INFO 521 Fall 2021, HW 2, Exercises 1 and 2
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 2
# of the current year (2021).
# You are NOT permitted to share this file with other students outside of
# this course year. Doing so will be considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

import os
import numpy
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_EXERCISE_1 = True
RUN_EXERCISE_2 = True


# -----------------------------------------------------------------------------
# Global paths -- You do NOT need to edit these paths!
# -----------------------------------------------------------------------------

DATA_ROOT = '../data'
DATA_WOMEN100 = os.path.join(DATA_ROOT, 'womens100.csv')
DATA_SYNTH = os.path.join(DATA_ROOT, 'synthdata2021.csv')

FIGURES_ROOT = '../figures'
PATH_TO_WOMENS100_FIG = os.path.join(FIGURES_ROOT, 'womens100-best-fit.png')
PATH_TO_SYNTH_FIG = os.path.join(FIGURES_ROOT, 'synthetic2021-3rd-poly.png')


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def read_data(filepath, d=','):
    """
    Wrapper to genfromtext a numpy.array of the data.

    :param filepath: sting representing path to data file to load.
    :param d: char string representing delimiter separating values on
              a line in data file.
    """
    return numpy.genfromtxt(filepath, delimiter=d, dtype=None)


def plot_data(x, t, title='Data'):
    """
    Plot single input feature x data with corresponding response
    values t as a scatter plot.

    :param x: 1-dimensional array of input data values (numbers)
    :param t: 1-dimensional array of responses
    :param title: title of plot (default 'Data')
    :return: None
    """
    plt.figure()  # Create a new figure object for plotting
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(title)
    plt.pause(.1)  # required on some systems to allow rendering


def plot_model(x, w, color='r'):
    """
    Plot the function (curve) of an n-th order polynomial model
    with one input (x):
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n
    This works by creating a set of x-axis (plotx) points and
    then uses the model parameters w to determine the corresponding
    t-axis (plott) points on the model curve.

    :param x: sequence of 1-dimensional input data values (numbers)
    :param w: n-dimensional sequence of model parameters: w0, w1, w2, ..., wn
    :param color: specifies the color of the model curve
    :return: the plotx and plott values for the plotted curve
    """
    # NOTE: this assumes a figure() object has already been created.

    # plotx represents evenly-spaced set of 100 points on the x-axis.
    # This is used for creating a relatively "smooth" model curve plot.
    # Includes points a little before the min x input (-0.25)
    # and a little after the max x input (+0.25)
    plotx = numpy.linspace(min(x)-0.25, max(x)+0.25, 100)

    # plotX (note that python is case sensitive, so this is *not*
    # the same as plotx with a lower-case x) is the "design matrix"
    # for our model curve inputs represented in plotx.
    # We need to do the same computation as we do when doing
    # model fitting (as in fitpoly(), below), except that we
    # don't need to infer (by the normal equations) the values
    # of w, as they are given here as input.
    # plotx.shape[0] ensures we create a matrix with the number of
    # rows corresponding to the number of points in plotx (this will
    # still work even if we change the number of plotx points to
    # something other than 100)
    plotX = numpy.zeros((plotx.shape[0], w.size))

    # populate the design matrix plotX
    for k in range(w.size):
        plotX[:, k] = numpy.power(plotx, k)

    # Take the dot (inner) product of the design matrix plotX and the
    # parameter vector w
    plott = numpy.dot(plotX, w)

    # plot the x (plotx) and t (plott) values in red
    plt.plot(plotx, plott, color=color, linewidth=2)

    plt.pause(.1)  # required on some systems to allow rendering
    return plotx, plott


def scale01(x):
    """
    HELPER FUNCTION: only needed if you are working with relatively
    large x values.

    NOTE: This is NOT needed for problems 1, 2 of hw2. It may be useful
    elsewhere.

    Mathematically, the sizes of the values of variables (such as x)
    could be arbitrarily large.  However, when we go to manipulate
    those values on a computer, we need to be careful.  E.g., in
    these exercises we are taking powers of values and if the
    values are large, then taking a large power of the variable
    may exceed what can be represented numerically in the computer.

    For example, in the Olympics data (both men's and women's),
    the input x values are years in the 1000's.  If your model
    is, say, polynomial order 5, then you're taking a large
    number to the power of 5, which is the order of a quadrillion!
    Python floating point numbers have trouble representing this
    many significant digits.

    This function here scales the input data to be the range [0, 1]
    (i.e., between 0 and 1, inclusive).

    :param x: sequence of 1-dimensional input data values (numbers)
    :return: x values linearly scaled to range [0, 1]
    """
    x_min = min(x)
    x_range = max(x) - x_min
    return (x - x_min) / x_range


# -------------------------------------------------------------------------
# fitpoly

def fitpoly(x, t, model_order, verbose_p=False):
    """
    Given "training" data in input array x (representing the
    features), corresponding target value array t, and a specified
    polynomial of order model_order, determine the linear
    least mean squared (LMS) error best fit for parameters w,
    using the generalized matrix normal equation.

    model_order is a non-negative integer, n, representing the
    highest polynomial exponent of the polynomial model:
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n

    :param x: sequence of 1-dimensional input data features
    :param t: sequence of target response values
    :param model_order: integer representing the highest polynomial
           exponent of the polynomial model
    :return: parameter vector w
    """

    # Construct the empty design matrix
    # numpy.zeros takes a python tuple representing the number
    # of elements along each axis (e.g., rows, columns) and returns
    # an array of those dimensions filled with zeros.
    # For example, to create a 2x3 array of zeros, call
    #     numpy.zeros((2,3))
    # and this returns (if executed at the command-line):
    #     array([[ 0.,  0.,  0.],
    #            [ 0.,  0.,  0.]])
    # The number of columns is model_order+1 because a model_order
    # of 0 requires one column (filled with input x values to the
    # power of 0 -- i.e., =1), model_order=1 requires two columns
    # (first input x values to power of 0, then column of input x
    # values to power 1), and so on...
    X = numpy.zeros((x.shape[0], model_order+1))

    # Fill each column of the design matrix with the corresponding
    # data taken to the power of the column-index
    for k in range(model_order+1):  # w.size
        X[:, k] = numpy.power(x, k)

    if verbose_p:
        print('model_order', model_order)
        print('x.shape', x.shape)
        print('X.shape', X.shape)
        print('t.shape', t.shape)

    #### YOUR CODE HERE ####
    a = numpy.linalg.inv(numpy.dot(X.T, X))
    b = numpy.dot(t.T,X)

    w = numpy.dot(a,b)
    
    if verbose_p:
        print('w.shape', w.shape)
    
    return w


# -----------------------------------------------------------------------------
# Helper script to run fitpoly on particular data set and plot data and model
# -----------------------------------------------------------------------------


def read_data_fit_plot(data_path, model_order, plot_title,
                       scale_p=False,
                       save_path=None,
                       verbose_p=False):
    """
    A "top-level" script to
        (1) Load the data
        (2) Optionally scale the data between [0, 1]
        (3) Plot the raw data
        (4) Find the best-fit parameters
        (5) Plot the model on top of the data
        (6) If save_path is a filepath (not None), then save the figure as a pdf
    :param data_path: Path to the data
    :param model_order: Non-negative integer representing model polynomial order
    :param scale_p: Boolean Flag (default False) If True, scales data between 0 and 1
    :param save_path: Optional (default None) filepath to save figure to file
    :param plot_p: Boolean Flag (default False)
    :param plot_title: Title of the plot (default 'Data')
    :param verbose_p: Boolean Flag (default False)
        If True, fitpoly call will print to stdout
    :return: None
    """

    print('\n-----------------------------------------------')

    # (1) load the data
    data = read_data(data_path, ',')

    # (2) Optionally scale the data between [0,1]
    # See the scale01 documentation for explanation of why you might want to scale
    if scale_p:
        x = scale01(data[:, 0])  # extract x (slice first column) and scale so x \in [0,1]
    else:
        x = data[:, 0]  # extract x (slice first column)
    t = data[:, 1]  # extract t (slice second column)

    # (3) plot the raw data
    plot_data(x, t, title=plot_title)

    # (4) find the best-fit model parameters using the fitpoly function
    w = fitpoly(x, t, model_order, verbose_p=verbose_p)

    print('Identified model parameters w (in scientific notation):\n', w)
    # python defaults to print floats in scientific notation,
    # so here I'll also print using python format, which I find easier to read
    print('w again (not in scientific notation):')
    print(' ', ['{0:f}'.format(i) for i in w])

    # (5) Plot the model on top of the data
    plot_model(x, w)

    # (6) If save_path is a filepath (not None), then save the figure as a pdf
    if save_path is not None:
        plt.savefig(save_path, format='png')

    return w


# -----------------------------------------------------------------------------
# Exercise 1
# -----------------------------------------------------------------------------

def exercise_1(data_path, figure_path):
    #### YOUR CODE HERE ####
    w =   read_data_fit_plot(data_path, 1, "fig_ex1", save_path=figure_path)

    return w


# -----------------------------------------------------------------------------
# Exercise 2
# -----------------------------------------------------------------------------

def exercise_2(data_path, figure_path):
    #### YOUR CODE HERE ####
    w = read_data_fit_plot(data_path, 3, "fig_ex2", save_path=figure_path)

    return w


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_EXERCISE_1:
        exercise_1(DATA_WOMEN100, PATH_TO_WOMENS100_FIG)
    if RUN_EXERCISE_2:
        exercise_2(DATA_SYNTH, PATH_TO_SYNTH_FIG)

    # call to the matplotlib.pyplot show() function to hold the figure plot window open
    plt.show()
