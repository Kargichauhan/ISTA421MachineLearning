import numpy
import os

import matplotlib.pyplot
from numpy import random

from numpy.lib.npyio import loadtxt



# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------

# Set each of the following lines to True in order to make the script execute
# each of the exercise functions

RUN_EXERCISE_6 = False
RUN_EXERCISE_7 = False
RUN_EXERCISE_8 = False
RUN_EXERCISE_9 = True


# -----------------------------------------------------------------------------
# Global paths
# -----------------------------------------------------------------------------

DATA_ROOT = '../data'
PATH_TO_WALK_DATA = os.path.join(DATA_ROOT, 'walk.txt')
PATH_TO_X_DATA = os.path.join(DATA_ROOT, 'X.txt')
PATH_TO_W_DATA = os.path.join(DATA_ROOT, 'w.txt')

FIGURES_ROOT = '../figures'
PATH_TO_WALK_FIGURE = os.path.join(FIGURES_ROOT, 'walk.png')


# -----------------------------------------------------------------------------
def walk_arr_scaled_fun(walk_arr, min_value, max_value):
    """
    This function is used to scale an array with the min-max normalization method.
    The limits on this function are [-1,1], therefore all the results of this function
    will be between those values.

    """
    #In order to scale, I used min-max normalization as shown in: https://en.wikipedia.org/wiki/Feature_scaling
    walk_arr_sca = []   #create the new array with all the new x
    for x in walk_arr:
        #[a,b] where a is the new min limit and b the new max limit
        #x_new = a + ((x-min(array))(b-a))/(max(array)-min(array))
        #since our limits are -1 and 1 
        # a = -1 and b = 1

        x_new = -1+((x-min_value)*(1-(-1)))/(max_value-min_value)   
        walk_arr_sca.append(x_new)  #insert x_new into the array
    return walk_arr_sca     #return the new scaled array

def exercise_6(path_to_data, path_to_figure):
    """
    Loading 1-d array of data from path_to_data, plot it
    and save it on path_to_figure.
    Then scaled the array and plot it again.

    """
    print('='*30)
    print('Running exercise_6()')

    #### YOUR CODE HERE ####
    walk_arr = numpy.loadtxt(path_to_data, delimiter= ",") #loads information from a file which is separate by commas


    #### YOUR CODE HERE ####
    # plot the data using matplotlib plot!
    matplotlib.pyplot.figure()              #use to to create a figure where all data is going to be display
    matplotlib.pyplot.plot(walk_arr)        #plot the data from the array created before
    matplotlib.pyplot.ylabel("Location")    #creates the y label
    matplotlib.pyplot.xlabel("Step")        #creates the x label
    matplotlib.pyplot.title("Random Walk")  #creates the title of the figure
    matplotlib.pyplot.savefig(path_to_figure)      #saves the figure as an image in figures file with the nomber walk.png
    
    print(f'walk_arr.shape: {walk_arr.shape}')

    #### YOUR CODE HERE ####
    walk_min = min(walk_arr)        #lowest value on random walk

    print(f'walk_min: {walk_min}')

    #### YOUR CODE HERE ####
    walk_max = max(walk_arr)        #highest value on random walk

    print(f'walk_max: {walk_max}')

    #### YOUR CODE HERE ####
    walk_arr_scaled = walk_arr_scaled_fun(walk_arr, walk_min, walk_max)

    print(walk_arr_scaled)
    matplotlib.pyplot.figure()              #use to to create a figure where all data is going to be display
    matplotlib.pyplot.plot(walk_arr_scaled)        #plot the data from the array created before

    print('DONE exercise_6()')

    return walk_arr, walk_min, walk_max, walk_arr_scaled


# -----------------------------------------------------------------------------

def exercise_7():
    """
    Usage of random seeds and random numbers.
    Calculates the probability of doubles with 2 dices.

    """
    print('=' * 30)
    print('Running exercise_7()')

    #### YOUR CODE HERE ####
    # set the numpy random seed to 7
    numpy.random.seed(7)
    # This determines how many times we "throw" the
    #   2 six-sided dice in an experiment
    num_dice_throws = 10000  # don't edit this!

    # This determines how many trials in each experiment
    #   ... that is, how many times we'll throw our two
    #   6-sided dice num_dice_throws times
    num_trials = 10  # don't edit this!

    # Yes, you can have functions inside of functions!
    
    def run_experiment():
        trial_outcomes = list()
        for trial in range(num_trials):
            #### YOUR CODE HERE ####
            doubles = 0             #keeps track of how many doubles do we have
            for i in range(num_dice_throws):    #loop for rolling a dice 
                x = random.randint(0,6)         #rolls dice 1     
                y = random.randint(0,6)         #rolls dice 2
                if(x == y):                     #if both are the same then +1 to doubles
                    doubles += 1
            
            # In the following, make it so that probability_estimate is an estimate
            # of the probability of throwing 'doubles' with two fair six-sided dice
            # (i.e., the probability that the dice end up with the same values)
            # based on throwing the two dice num_dice_throws times.
            probability_estimate = doubles/num_dice_throws 

            # Save the probability estimate for each trial (you don't need to change
            # this next line)
            trial_outcomes.append(probability_estimate)
        return trial_outcomes

    experiment_outcomes_1 = run_experiment()

    print(f'experiment_outcomes_1: {experiment_outcomes_1}')

    print(f'do it again!')

    experiment_outcomes_2 = run_experiment()
    print(f'experiment_outcomes_2: {experiment_outcomes_2}')

    print('Now reset the seed')

    #### YOUR CODE HERE ####
    # reset the numpy random seed back to 7
    numpy.random.seed(7)

    experiment_outcomes_3 = run_experiment()

    print(f'experiment_outcomes_3: {experiment_outcomes_3}')

    print("DONE exercise_7()")

    return experiment_outcomes_1, experiment_outcomes_2, experiment_outcomes_3


# -----------------------------------------------------------------------------

def exercise_8():
    """
    Working with matrixes and some linear algebra. 
    Also creating random matrices with a random seed.

    """

    print("=" * 30)
    print("Running exercise_8()")

    #### YOUR CODE HERE ####
    # set the numpy random seed to 7
    numpy.random.seed(7)
    #### YOUR CODE HERE ####
    # Set x to a 2-d array of random number of shape (3, 1)

    #https://numpy.org/doc/1.16/reference/generated/numpy.random.rand.html#numpy.random.rand
    x = numpy.random.rand(3,1) #creates a 2-d array of random numbers

    print(f'x:\n{x}')

    #### YOUR CODE HERE ####
    # Set y to a 2-d array of random number of shape (3, 1)
    y = numpy.random.rand(3,1)

    print(f'y:\n{y}')

    #### YOUR CODE HERE ####
    # Calclate the sum of x and y
    v1 = x+y

    print(f'v1:\n{v1}')

    #### YOUR CODE HERE ####
    # Calclate the multiplication of x and y

    # for matrices it can used the * to element-wise multiply 
    # https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy

    v2 = x*y

    print(f'v2:\n{v2}')

    #### YOUR CODE HERE ####
    # Transpose x

    #https://www.geeksforgeeks.org/transpose-matrix-single-line-python/
    #Numpy contanins function where is possibly to do the transpose either with 
    #numpy.transpose(matrix)  or  matrix.T
    xT = x.T

    print(f'xT: {xT}')

    #### YOUR CODE HERE ####
    # Calculate the dot product of x and y
    # numpy contains also a function where you can get th dot product of two matrices
    # https://numpy.org/doc/stable/reference/generated/numpy.dot.html

    v3 = numpy.dot(xT,y)

    print(f'v3: {v3}')

    #### YOUR CODE HERE ####
    # Set A to a 2-d array of random numbers of shape (3, 3)
    A = numpy.random.rand(3,3)

    print(f'A:\n{A}')

    #### YOUR CODE HERE ####
    # Compute the dot product of x-transpose with A
    v4 = numpy.dot(xT,A)

    print(f'v4: {v4}')

    #### YOUR CODE HERE ####
    # Compute the dot product of x-transpose with A and the product with y
    v5 = numpy.dot(v4,y)

    print(f'v5: {v5}')

    #### YOUR CODE HERE ####
    # Compute the inverse of A

    #https://www.tutorialspoint.com/numpy/numpy_inv.htm 
    v6 = numpy.linalg.inv(A) 

    print(f'v6:\n{v6}')

    #### YOUR CODE HERE ####
    # Compute the dot product of A with its inverse.
    #   Should be near identity (save for some numerical error)
    v7 = numpy.dot(v6,A)

    print(f'v7:\n{v7}')

    return x, y, v1, v2, xT, v3, A, v4, v5, v6, v7


# -----------------------------------------------------------------------------

def exercise_9(path_to_X_data, path_to_w_data):
    """
    Compares the difference between implementing the same calculation
    using scalar and then vector math in code.

    """

    print("="*30)
    print("Running exercise_9()")

    #### YOUR CODE HERE ####
    # load the X and w data from file into arrays
    X = numpy.loadtxt(path_to_X_data,delimiter=",")
    w = numpy.loadtxt(path_to_w_data)

    print(f'X:\n{X}')
    print(f'w: {w}')

    #### YOUR CODE HERE ####
    # Extract the column 0 (x_n1) and column 1 (x_n2) vectors from X
    x_n1 = X[:,0]
    x_n2 = X[:,1]

    print(f'x_n1: {x_n1}')
    print(f'x_n2: {x_n2}')

    #### YOUR CODE HERE ####
    # Use scalar arithmetic to compute the right-hand side of Exercise 3
    #   (Exercise 1.3 from FCMA p.35)
    # Set the final value to
    sum_xn1 = 0
    sum_xn2 = 0
    sum_xn1_xn2 = 0
    mult_xn1_xn2 = x_n1*x_n2
    for i in x_n1:
        sum_xn1+=(i**2)
    for i in x_n2:
        sum_xn2+=(i**2)
    for i in mult_xn1_xn2:
        sum_xn1_xn2+=i;

    scalar_result = ((w[0]**2)*sum_xn1)+2*w[0]*w[1]*sum_xn1_xn2+((w[1]**2)*sum_xn2)

    print(f'scalar_result: {scalar_result}')

    #### YOUR CODE HERE ####
    # Now you will compute the same result but using linear algebra operators.
    #   (i.e., the left-hand of the equation in Exercise 1.3 from FCMA p.35)
    # You can compute the values in any linear order you want (but remember,
    # linear algebra is *NOT* commutative!), however here will require you to
    # first computer the inner term: X-transpose times X (XX), and then
    # below you complete the computation by multiplying on the left and right
    # by w (wXXw)
    XX = numpy.dot((numpy.transpose(X)),X)

    print(f'XX:\n{XX}')

    #### YOUR CODE HERE ####
    # Now you'll complete the computation by multiplying on the left and right
    # by w to determine the final value: wXXw
    wXXw = numpy.dot(w.T,numpy.dot(XX,w))

    print(f'wXXw: {wXXw}')

    print("DONE exercise_9()")

    return X, w, x_n1, x_n2, scalar_result, XX, wXXw


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if RUN_EXERCISE_6:
        exercise_6(PATH_TO_WALK_DATA, PATH_TO_WALK_FIGURE)
        #### YOUR CODE HERE ####
        # Add a call to the matplotlib.pyplot show() function
        matplotlib.pyplot.show()
    if RUN_EXERCISE_7:
        exercise_7()
    if RUN_EXERCISE_8:
        exercise_8()
    if RUN_EXERCISE_9:
        exercise_9(PATH_TO_X_DATA, PATH_TO_W_DATA)
