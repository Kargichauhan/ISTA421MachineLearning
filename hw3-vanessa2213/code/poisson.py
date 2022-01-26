# ISTA 421 / INFO 521 Fall 2021, HW 3, Exercise 1
# Author: Clayton T. Morrison
# This file is only made available for use in your submission for Homework 3
# of the current year (2021).
# You are NOT permitted to share this file with other students outside of
# this course year. Doing so will be considered cheating by you and others not
# in the current course year. Cheating can be assessed retroactively, even
# after you graduate.

import math 

# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions


RUN_EX1_A = True
RUN_EX1_B = True


# -----------------------------------------------------------------------------
# Exercises
# -----------------------------------------------------------------------------


def calculate_poisson_pmf_a():
    """
    Calculate probability that Y ~ Poisson(lambda=5) for 3 <= Y <= 7
    :return: probability
    """
    ### YOUR CODE HERE
    probability = 0  # NOTE: 0 is not the correct answer!
    Yminor = 3
    Ymax = 7
    landa = 5

    for y in range(Yminor,Ymax+1):  #a loop to sum all the probabilities from 3 to 7
        probability += (landa**y/math.factorial(y))*math.exp(-landa)
        #print(probability)
    
    #print(probability)
    return probability
    


def calculate_poisson_pmf_b():
    """
    Calculate probability that Y ~ Poisson(lambda=5) for Y < 3 or Y > 7
    :return: probability
    """
    ### YOUR CODE HERE
    probability = 0 #  NOTE: 0 is not the correct answer!
    probability = 1-calculate_poisson_pmf_a()   #the probability of Y<3 or Y<7 is 1- (Y>=3 & Y<=7 )
    #print(probability)
    return probability


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_EX1_A:
        calculate_poisson_pmf_a()
    if RUN_EX1_B:
        calculate_poisson_pmf_b()
