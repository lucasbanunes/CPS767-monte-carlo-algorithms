import numpy as np


def get_stationary_distribution(markov_matrix):
    n_states = markov_matrix.shape[0]
    # Solve the equation πP = π
    # where sum(π) = 1
    A = np.vstack((markov_matrix.T - np.eye(n_states), np.ones((1, n_states))))
    b = np.zeros(n_states + 1)
    b[-1] = 1
    return np.linalg.lstsq(A, b, rcond=0)


def get_spectral_gap(markov_matrix):
    # Get the eigenvalues of the Markov matrix
    eigenvalues, eigenvectors = np.linalg.eig(markov_matrix)
    sortidx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[sortidx]
    eigenvectors = eigenvectors[:, sortidx]
    abs_eigenvalues = np.abs(eigenvalues)
    gap = 1 - abs_eigenvalues[1]
    return gap, eigenvalues, eigenvectors


def total_variation(u, v):
    return .5*np.sum(np.abs(u-v))


def get_mixture_time(epsilon,
                     transition_matrix,
                     stationary_distribution):
    n_states = transition_matrix.shape[0]
    state_matrix = np.eye(n_states)
    current_variation = np.full(n_states, np.inf)
    transition_counter = 0
    stationary_distribution = stationary_distribution.flatten()
    stationary_matrix = [
        stationary_distribution for _ in range(n_states)
    ]
    stationary_matrix = np.array(stationary_matrix)
    current_variation = 0.5*np.sum(
        np.abs(state_matrix - stationary_matrix),
        axis=1
    )
    max_variations = [max(current_variation)]
    while np.all(current_variation > epsilon) and (transition_counter <= 9999):
        # Compute the variation distance
        # between the current state and the stationary distribution
        transition_counter += 1
        state_matrix = state_matrix @ transition_matrix
        current_variation = 0.5*np.sum(
            np.abs(state_matrix - stationary_matrix),
            axis=1
        )
        max_variations.append(max(current_variation))

    if transition_counter > 9999:
        print("Warning: The Markov chain did not converge"
              " within 10000 iterations.")
        return None

    return transition_counter, np.array(max_variations)
