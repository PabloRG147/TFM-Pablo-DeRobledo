import numpy as np


def kullback_leibler_divergence(dist_prob1: dict, dist_prob2: dict) -> float:
    """Computes Kullback Leibler divergence 
    between two probability distributions

    Args:
        dist_prob1 (dict): A dictionary with
        the probability distribution of bitstrings.
        dist_prob2 (dict): A dictionary with
        the probability distribution of bitstrings.

    Returns: The value of Kullback Leibler divergence between
    the probabilities distributions given as input
    """

    # Checks if the sum of dist_prob1 values is equal to 1.
    if not np.isclose(sum(dist_prob1.values()), 1):
        raise ValueError("The sum of dist_prob1 values is not equal to 1!")

    # Checks if the sum of dist_prob2 values is equal to 1.
    if not np.isclose(sum(dist_prob2.values()), 1):
        raise ValueError("The sum of dist_prob2 values is not equal to 1!")

    # Creates the intersection of bitstrings from inputs distributions.
    bitstrings1 = set(dist_prob1.keys())
    bitstrings2 = set(dist_prob2.keys())
    intersection = list(bitstrings1.intersection(bitstrings2))

    # Computing KL divergence
    kl_div = np.array(
        [
            (
                dist_prob1[x] * np.log(dist_prob1[x] / dist_prob2[x])
                if not (np.isclose(dist_prob1[x], 0) and np.isclose(dist_prob2[x], 0))
                else 0
            )
            for x in intersection
        ]
    ).sum()

    return kl_div


def shannon_entropy(dist_prob: dict) -> float:
    """Computes the Shannon entropy of a
    probability distribution.
    
    Args:
        dist_prob (dict): A dictionary with
        the probability distribution of bitstrings.

    Returns: The Shannon entropy of the given
    probability distribution.
    """

    # Checks if the sum of dist_prob values is equal to 1.
    if not np.isclose(sum(dist_prob.values()), 1):
        raise ValueError("The sum of dist_prob values is not equal to 1!")

    # Computes the Shannon entropy.
    shannon_entropy = (-1) * np.array(
        [
            (
                dist_prob[x] * np.log2(dist_prob[x])
                if not np.isclose(dist_prob[x], 0)
                else 0
            )
            for x in dist_prob.keys()
        ]
    ).sum()

    return shannon_entropy
