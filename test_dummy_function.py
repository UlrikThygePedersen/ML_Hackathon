from dummy_function import multiply_by_two

def test_multiply_by_two_positive_number():
    assert multiply_by_two(3) == 6

def test_multiply_by_two_zero():
    assert multiply_by_two(0) == 0

def test_multiply_by_two_negative_number():
    assert multiply_by_two(-4) == -8