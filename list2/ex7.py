# Estimating the number of domains under the ufrj.br name of size <= k
# that use only ascii_lowercase letters.

from typing import Generator
from itertools import product
import string
import numpy as np
import socket


def domain_generator(k: int) -> Generator[str, None, None]:
    """
    Generates all possible domain names of size <= k under the ufrj.br domain.
    """

    # Generate all combinations of letters and digits for the domain name
    for length in range(1, k + 1):
        for name in product(string.ascii_lowercase, repeat=length):
            yield ''.join(name) + '.ufrj.br'


def domain_exists(domain: str) -> bool:
    """
    Checks if a domain exists in the DNS.
    """
    try:
        # Checks if the domain exists by trying to resolve it
        # Raises socket.gaierror if the domain does not exist
        socket.getaddrinfo(domain, 0)
        return True
    except socket.gaierror:
        return False


def sum_of_powers(bases: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """
    Returns the sum of the powers of the bases raised to the exponents.
    """
    return np.sum(np.power(bases, exponents))


def number_of_possible_domains(k: int) -> int:
    """
    Returns the number of possible domain names of size <= k
    under the ufrj.br domain.
    """
    return sum_of_powers(
        np.full(k, len(string.ascii_lowercase)),
        np.arange(1, k+1)
    )


def monte_carlo_estimate(k: int, n: int) -> float:
    """
    Estimates the number of domains under the ufrj.br name of size <= k
    using Monte Carlo estimation.
    """
    x = np.random.uniform(0, 1, n)
    bases = np.full(
        len(x),
        len(string.ascii_lowercase)
    )
    # No need to divide and multiply by k to
    # avoid numerical instability
    estimate = sum_of_powers(bases, x)
    expected_value = estimate/k
    return estimate, expected_value
