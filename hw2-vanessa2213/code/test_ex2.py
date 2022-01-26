from code import fitpoly
import numpy
import pytest
import os


@pytest.fixture
def exercise_results():
    return fitpoly.exercise_2('data/synthdata2021.csv', 'figures/synthetic2021-3rd-poly.png')


def test_ex2_w(exercise_results):
    target = numpy.array([18.27934516, 7.58594459, -3.70990554, -0.94480755])
    w = exercise_results
    assert numpy.allclose(w, target)


def test_synthetic_poly_figure_exists():
    assert os.path.exists('figures/synthetic2021-3rd-poly.png')
