# ISTA 421 / INFO 521 Fall 2021, HW 3, Exercise 2
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 3
# of the current year (2021).
# You are NOT permitted to share this file with other students outside of
# this course year. Doing so will be considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

# approx_expected_value_sin.py
# Port of approx_expected_value.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Approximating expected values via sampling
import numpy
import os
import matplotlib.pyplot as plt
import random_generator


# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_DEMO = False
RUN_EXERCISE_2 = True


# -----------------------------------------------------------------------------
# Global paths
# -----------------------------------------------------------------------------

DATA_ROOT = os.path.join('..', 'data')
PATH_TO_RANDOM_UNIFORM_10000 = os.path.join(DATA_ROOT, 'rand_uniform_10000.txt')

FIGURES_ROOT = os.path.join('..', 'figures')
PATH_TO_FN_APPROX_FIG = os.path.join(FIGURES_ROOT, 'ex2_fn_approx.png')


# -----------------------------------------------------------------------------
# Sampling demo code
# -----------------------------------------------------------------------------

# We are trying to estimate the expected value of
# $f(y) = y^2$
##
# ... where
# $p(y)=U(0,1)$
##
# ... which is given by ('\int' is the latex code for "integral"):
# $\int y^2 p(y) dy$
##
# The exact result is:
# $\frac{1}{3}$ = 0.333...
# (NOTE: this just gives you the result -- you should be able to derive it!)

# First, let's plot the function, shading the area under the curve for x=[0,1]
# The following plot_fn helps us do this.

# Information about font to use when plotting the function
FONT = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 12,
        }


def plot_fn(fn, a, b, fn_name=None, resolution=100):
    """
    Plots a function fn between lower bound a and upper bound b.
    :param fn: a function of one variable
    :param a: lower bound
    :param b: upper bound
    :param fn_name: the name of the function (displayed in the plot)
    :param resolution: the number of points between a and b used to plot the fn curve
    :return: None
    """
    x = numpy.append(numpy.array([a]),
                     numpy.append(numpy.linspace(a, b, resolution), [b]))
    y = fn(x)
    y[0] = 0
    y[-1] = 0
    plt.figure()
    plt.fill(x, y, 'b', alpha=0.3)
    if fn_name:
        fname, x_tpos, y_tpos = fn_name()
        plt.text(x_tpos, y_tpos, fname, fontdict=FONT)
    plt.title('Area under function')
    #plt.xlabel('$y$') #for demo only
    #plt.ylabel('$f(y)$') #for demo only
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')

    x_range = b - a

    plt.xlim(a-(x_range*0.1), b+(x_range*0.1))


# Define the function y that we are going to plot
def y_fn(x):
    """
    The function x^2
    :param x:
    :return:
    """
    #return numpy.power(x, 2) #for the demo only
    return 35+3*x-0.5*numpy.power(x, 3)+0.05*numpy.power(x, 4)


def y_fn_name():
    """
    Helper for displaying the name of the fn in the plot
    Returns the parameters for plotting the y function name,
    used by approx_expected_value
    :return: fname, x_tpos, y_tpos
    """
    #fname = r'$f(y) = y^2$'  # latex format of fn name for demo
    fname = r'$f(x) = 35+3x-0.5x^3+0.05x^4$'  # latex format of fn 
    x_tpos = 0.1  # x position for plotting text of name
    y_tpos = 0.5  # y position for plotting text of name
    return fname, x_tpos, y_tpos


def run_sample_demo():
    # Plot the function!
    plot_fn(y_fn, 0, 1, y_fn_name)

    # Now we'll approximate the area under the curve using sampling...

    # Sample 1000 uniformly random values in [0..1]
    ys = numpy.random.uniform(low=0.0, high=1.0, size=1000)
    # compute the expectation of y, where y is the function that squares its input
    ey2 = numpy.mean(numpy.power(ys, 2))
    print('\nSample-based approximation: {:f}'.format(ey2))

    # Store the evolution of the approximation, every 10 samples
    sample_sizes = numpy.arange(1, ys.shape[0], 10)
    ey2_evol = numpy.zeros((sample_sizes.shape[0]))  # storage for the evolving estimate...
    # the following computes the mean of the sequence up to i, as i iterates
    # through the sequence, storing the mean in ey2_evol:
    for i in range(sample_sizes.shape[0]):
        ey2_evol[i] = numpy.mean(numpy.power(ys[0:sample_sizes[i]], 2))

    # Create plot of evolution of the approximation
    plt.figure()
    # plot the curve of the estimation of the expected value of f(x)=y^2
    plt.plot(sample_sizes, ey2_evol)
    # The true, analytic result of the expected value of f(y)=y^2 where y ~ U(0,1): $\frac{1}{3}$
    # plot the analytic expected result as a red line:
    plt.plot(numpy.array([sample_sizes[0], sample_sizes[-1]]),
             numpy.array([1./3, 1./3]), color='r')
    plt.xlabel('Sample size')
    plt.ylabel('Approximation of expectation')
    plt.title('Approximation of expectation of $f(y) = y^2$')
    plt.pause(.1)  # required on some systems so that rendering can happen


# -----------------------------------------------------------------------------
# Exercise 2
# -----------------------------------------------------------------------------
def exercise_2(path_to_random_numbers, figure_path):
    """
    Provide code to
    :param path_to_random_numbers: directory path to random number source file
    :return: expected, estimate
    """

    # Do not edit the following.
    # Use the following random number generator (rng) to draw random samples
    # This will allow you to generate Uniform random samples (this will load
    # the random numbers from file, and they're already set up to be in the
    # uniform random range for this assignment)
    rng = random_generator.RNG(path_to_random_numbers)

    # In your code, rather than calling
    #     numpy.random.uniform(low, high, size)
    # instead use the following function to draw n (i.e., size) samples.
    # Like numpy.random.uniform, this will return an array of uniform random
    # numbers, of length n:
    #     rng.get_n_random(n)

    #### YOUR CODE HERE ####
    expected = 0  # Calculate the expected value; 0 is not the right answer!
    estimate = 0  # Calculate the sample approximation of the expected value; 0 is not the right answer!
    a= -4
    b=10
    #integral of f(x)p(x)dx 
    expected = (1/(b-a))*((35*b+1.5*numpy.power(b,2)-0.125*numpy.power(b,4)+0.01*numpy.power(b,5))-(35*a+1.5*numpy.power(a,2)-0.125*numpy.power(a,4)+0.01*numpy.power(a,5)))
    print(f'Expectation: {expected}')

    #plot_fn(y_fn, a, b, y_fn_name)

    n= 10000
    xs = rng.get_n_random(n)

    estimate = numpy.mean(35+3*xs-0.5*numpy.power(xs, 3)+0.05*numpy.power(xs, 4))
    print(f'Sample-based approximation: {estimate}')
    
    # Store the evolution of the approximation, every 10 samples
    sample_sizes = numpy.arange(1, xs.shape[0], 10)
    ey2_evol = numpy.zeros((sample_sizes.shape[0]))  # storage for the evolving estimate...
    # the following computes the mean of the sequence up to i, as i iterates
    # through the sequence, storing the mean in ey2_evol:
    for i in range(sample_sizes.shape[0]):
        #ey2_evol[i] = numpy.mean(numpy.power(xs[0:sample_sizes[i]], 2))
        ey2_evol[i] = numpy.mean(35+3*xs[0:sample_sizes[i]]-0.5*numpy.power(xs[0:sample_sizes[i]], 3)+0.05*numpy.power(xs[0:sample_sizes[i]], 4))

    # Create plot of evolution of the approximation
    plt.figure()
    # plot the curve of the estimation of the expected value of f(x)=y^2
    plt.plot(sample_sizes, ey2_evol)
    # The true, analytic result of the expected value of f(y)=y^2 where y ~ U(0,1): $\frac{1}{3}$
    # plot the analytic expected result as a red line:
    plt.plot(numpy.array([sample_sizes[0], sample_sizes[-1]]),
             numpy.array([expected, expected]), color='r')
    plt.xlabel('Sample size')
    plt.ylabel('Approximation of expectation')
    plt.title('Approximation of expectation of $f(x) = 35+3x-0.5x^3+0.05x^4$')
    plt.pause(.1)  # required on some systems so that rendering can happen
    if figure_path is not None:
        plt.savefig(figure_path)
  

    
    

    return expected, estimate



# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_DEMO:
        run_sample_demo()
        plt.show()
    if RUN_EXERCISE_2:
        exercise_2(PATH_TO_RANDOM_UNIFORM_10000, PATH_TO_FN_APPROX_FIG)
        plt.show()
