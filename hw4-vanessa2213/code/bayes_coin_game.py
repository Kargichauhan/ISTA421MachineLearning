# ISTA 421 / INFO 521 Fall 2021, HW 4, Exercise
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 3
# of the current year (2021).
# You are NOT permitted to share this file with other students outside of
# this course year. Doing so will be considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

# NOTE: You are only allowed to use the following imports.
#       You may not import any other modules / functions.

from scipy.special import gamma      # The Gamma function
from scipy.special import loggamma   # The log of the Gamma function
from scipy.special import binom      # binomial coefficient; i.e.: number of combinations, AKA "N choose K"
import numpy                         # NOTE: includes numpy.log and numpy.exp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

def plot_densities(r_values, prior, posterior, title: str):
    """
    Helper function to plot the prior and posterior
    :param r_values: r values (between 0 and 1) -- determines the x-axis values
    :param prior: Prior density values -- determines y-axis values
    :param posterior: Posterior density values -- determines y-axis values
    :param title: Title of the figure (Used to indicate the Scenario)
    :return:
    """
    plt.figure()
    plt.plot(r_values, prior, label='prior')
    plt.plot(r_values, posterior, label='posterior')
    plt.xlabel('$r$')
    plt.ylabel('$p(r)$')
    plt.legend(loc="upper left")
    plt.title(title)
    filename_base = '_'.join(title.split(' '))
    plt.savefig(f'../figures/{filename_base}.png', format='png')


def calculate_prior_density(r, alpha: float, beta: float):
    """
    Calculates the prior density of r for the coin game, as a function of
    the prior belief about the number of heads (alpha) and number of tails
    (beta).
    You can implement this a scalar fn returning a single scalar density,
    or as a vectorized function (where r is a numpy array) returning an
    array of densities.
    :param r: scalar or vector representation of values of r for which to compute the beta density
    :param alpha: Prior for heads
    :param beta: Prior for tails
    :return: The beta density at r (either scalar or vector)
    """
    ### Insert your code here ###
    density = (gamma(alpha+beta)/(gamma(alpha)*gamma(beta)))*pow(r,alpha-1)*pow(1-r,beta-1) # NOTE: 0 is NOT the solution!
    return density


def calculate_posterior_density(r, n: int, y_obs: int, alpha: float, beta: float):
    """
    Calculates the posterior density as a function of the prior and likelihood.
    You can implement this a scalar fn returning a single scalar posterior density,
    or as a vectorized function (where r is a numpy array) returning an array of
    densities.
    :param r: Scalar or vector representation of values of r for which to compute the beta density
    :param n: Total number of observations (how many coin flips)
    :param y_obs: Number of observed heads
    :param alpha: Prior for heads
    :param beta: Prior for tails
    :return: The beta density at r (either scalar or vector)
    """
    ### Insert your code here ###
    density = gamma(alpha+beta+n)/(gamma(alpha+y_obs)*gamma(beta+n-y_obs))*pow(r,alpha+y_obs-1)*pow(1-r,beta+n+-y_obs-1)  # NOTE: 0 is NOT the solution!
    return density


def calculate_marginal_likelihood(n: int, y_obs: int, alpha: float, beta: float) -> float:
    """
    Calculates the marginal likelihood of the data as a function of the
    prior parameterization.
    NOTE: Due to the potential for numerical underflow while performing the
    calculation, it is recommended to calculate the marginal likelihood in log-space
    (i.e., take the log of the equation used to compute the marginal likelihood)
    and then take the exponential of the result (numpy.exp(result)) to get back the
    probability density.
    :param n: Total number of observations (how many coin flips)
    :param y_obs: Number of observed heads
    :param alpha: Prior for heads
    :param beta: Prior for tails
    :return:
    """
    ### Insert your code here    ###
    marginal_likelihood = numpy.exp(numpy.log(binom(n,y_obs))+loggamma(alpha+beta)+loggamma(alpha+y_obs)+loggamma(beta+n-y_obs)-loggamma(alpha)-loggamma(beta)-loggamma(alpha+beta+n)) # NOTE: 0 is NOT the solution!
    return marginal_likelihood


def calculate_probability_of_winning(n: int, y_obs: int, alpha: float, beta: float) -> float:
    """
    Calculates the probability of winning given the observed data and the
    priors.
    NOTE: Due to the potential for numerical underflow while performing the
    calculation, it is recommended to calculate the marginal likelihood in
    log-space (i.e., take the log of the equation used to compute the marginal
    likelihood) and then take the exponential of the result (numpy.exp(result))
    to get back the probability density.
    :param n:
    :param y_obs:
    :param alpha:
    :param beta:
    :return:
    """
    ### Insert your code here ###
    sum = 0
    n_new =10
    d = alpha+y_obs
    g = beta+(n-y_obs)
    

    for i in range(7,n_new+1):
        sum+=numpy.exp(numpy.log(binom(n_new,i))+loggamma(d+g)+loggamma(d+i)+loggamma(g+n_new-i)-loggamma(d)-loggamma(g)-loggamma(d+g+n_new)) # NOTE: 0 is NOT the solution!
   


    probability_of_winning = 1-sum  # NOTE: 0 is NOT the solution!
    return probability_of_winning


def run_scenario(n: int, y_obs: int, alpha: float, beta: float, title: str, plot_p: bool=True):
    """
    Script to calculate the prior and posterior densities, the marginal
    likelihood of the data, and the probability of winning the game (assuming
    you need 6 or fewer heads out of 10 tosses) given prior beliefs about
    heads and tials, and the observed data (parameterized by the total number
    of coin flips, n, and the number of heads, y_obs).
    r_values represents the values of r for which you'll estimate the prior and
    posterior density values (for purposes of plotting).
    You must fill in the calculation of the r_prior, r_posterior,
    marginal_likelihood and probability_of_winning.

    NOTE: As the initial "dummy" values for r_prior and r_posterior are intended to
    suggest, it is required that both variables will be numpy arrays (of length 100)
    that contain the prior and posterior densities corresponding to the r_values.
    These are to be calculated by the functions prior_density and posterior_density,
    which you will need to implement.
    It is up to you whether you implement those functions as scalar functions
    (in which case you must then fill the r_prior and r_posterior arrays in a loop)
    OR as vectorized functions (that directly compute the array of r_prior and
    r_posterior values, respectively).

    :param alpha: Prior belief about prevalence of heads ("virtual evidence" of number of heads)
    :param beta: Prior belief about prevalence of tails ("virtual evidence" of number of tails)
    :param n: Total number of observations
    :param y_obs: Total number of observed heads
    :param title: Title of scenario
    :param plot_p: Flag controlling whether to plot; default True
    :return:
    """
    print(f'Calculating beliefs for {title}')
    r_values = numpy.linspace(start=0, stop=1, num=100)

    ### Insert your code here, computing the following four variables ###
    r_prior = calculate_prior_density(r_values,alpha,beta) # numpy array to be set by the function beta_density
    r_posterior = calculate_posterior_density(r_values,n,y_obs,alpha,beta)  # numpy array to be set by the function calculate_posterior
    marginal_likelihood = calculate_marginal_likelihood(n,y_obs,alpha,beta)  # implement calculate_marginal_likelihood and call it
    probability_of_winning = calculate_probability_of_winning(n,y_obs,alpha,beta)  # implement calculate_probability_of_winning and call it
    ### END of your code

    #if plot_p:
    #    plot_densities(r_values, r_prior, r_posterior, title=title)
    print(f'marginal_likelihood: {marginal_likelihood}')
    print(f'probability_of_winning: {probability_of_winning}')

    return r_prior, r_posterior, marginal_likelihood, probability_of_winning


if __name__ == "__main__":
    """
    Top-level script.
    """

    # The evidence:
    n = 20      # total observations (total number of coin flips)
    y_obs = 14  # number of observed heads

    # Scenarios - differences in prior beliefs
    run_scenario(n=n, y_obs=y_obs, alpha=1, beta=1, title='Scenario 1')
    run_scenario(n=n, y_obs=y_obs, alpha=50, beta=50, title='Scenario 2')
    run_scenario(n=n, y_obs=y_obs, alpha=5, beta=1, title='Scenario 3')

    # Holds execution to permit observing generated plots
    plt.show()
