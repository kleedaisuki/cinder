from cinder import add


def test_add_returns_sum() -> None:
    assert add(2, 3) == 5


def test_add_accepts_negative_values() -> None:
    assert add(-4, 6) == 2
