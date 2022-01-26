

# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set the following to True in order to make the script execute
# the exercise function
import random
import math
RUN_PI_ESTIMATE = True


# -----------------------------------------------------------------------------
# Estimate pi
# -----------------------------------------------------------------------------

def estimate_pi(N):
    """
    Calculate an estimate of pi (i.e., DO NOT use math.pi or any other direct
    representation of pi) by sampling, when only provided the ability to sample
    from a uniform distribution (using random.uniform) and the fact that area,
    A, of a circle with radius, r, is defined by: A = pi * r^2
    :param N: Number of samples
    :return: estimate of pi
    """

    ### YOUR CODE HERE
    estimated_pi = 0  # Calculate pi! pi != 0
    points_inside_circle = 0
    r = 1000000
    for i in range(N):
        x = random.uniform(-r,r)
        y = random.uniform(-r,r)
        if((x*x + y*y)<=r*r):
            points_inside_circle+=1
    
    estimated_pi = points_inside_circle/(math.pow(0.5,2)*N)


    print(estimated_pi)
    return estimated_pi


# -----------------------------------------------------------------------------
# TOP LEVEL SCRIPT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_PI_ESTIMATE:
        estimate_pi(1000000)

