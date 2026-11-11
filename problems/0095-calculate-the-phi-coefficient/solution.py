def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.
    """
    n11 = n00 = n10 = n01 = 0

    for a, b in zip(x, y):
        if a == 1 and b == 1:
            n11 += 1
        elif a == 0 and b == 0:
            n00 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1

    numerator = (n11 * n00) - (n10 * n01)

    denominator = (
        (n10 + n11) *
        (n00 + n01) *
        (n01 + n11) *
        (n00 + n10)
    ) ** 0.5

    if denominator == 0:
        return 0.0

    val = numerator / denominator

    return round(val, 4)