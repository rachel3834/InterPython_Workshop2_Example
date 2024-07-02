"""Tests for statistics functions within the Model layer."""

import pandas as pd
import numpy as np
import pytest

@pytest.mark.parametrize(
    "test_df, test_colname, expected",
    [
        (pd.DataFrame(data=[[1, 5, 3], 
                            [7, 8, 9], 
                            [3, 4, 1]], 
                      columns=list("abc")),
        "a",
        7),
        (pd.DataFrame(data=[[0, 0, 0], 
                            [0, 0, 0], 
                            [0, 0, 0]], 
                      columns=list("abc")),
        "b",
        0),
    ])
def test_max_mag(test_df, test_colname, expected):
    """Test max function works for array of zeroes and positive integers."""
    from lcanalyzer.models import max_mag
    assert max_mag(test_df, test_colname) == expected

@pytest.mark.parametrize(
    "test_input, test_input_colname, expected",
    [
        (
            np.random.normal(loc=10.0, scale=0.1, size=(3,3)),
            'b',
            10.0
        )
    ]
)
def test_mean_mag(test_input, test_input_colname, expected):
    # Test that mean_mag function works for ones
    from lcanalyzer.models import mean_mag

    test_input_df = pd.DataFrame(data=test_input, columns=list("abc"))

    np.testing.assert_allclose(mean_mag(test_input_df, test_input_colname), expected, rtol=0.1)

@pytest.mark.parametrize(
    "test_input, test_input_colname, expected",
    [
        (
            np.ones((3,3)),
            "a",
            1.0
        )
    ]
)
def test_min_mag(test_input, test_input_colname, expected):
    # Test that max_mag function works for integers
    from lcanalyzer.models import min_mag

    test_input_df = pd.DataFrame(data=np.ones((3,3)), columns=list("abc"))

    assert min_mag(test_input_df, test_input_colname) == expected

@pytest.mark.parametrize(
    "test_input, test_input_colname, expected",
    [
        ("string", "b", TypeError)
    ]
)
def test_max_mag_strings(test_input, test_input_colname, expected):
    # Test for TypeError when passing a string
    from lcanalyzer.models import max_mag

    with pytest.raises(expected):
        error_expected = max_mag(test_input, test_input_colname)