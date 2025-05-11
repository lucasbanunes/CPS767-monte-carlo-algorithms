# Source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

import numpy as np
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", fontsize='medium', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  size=fontsize,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels,
                  size=fontsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def get_stationary_distribution(transition_matrix):
    n_states = transition_matrix.shape[0]
    # Solve the equation πP = π
    # where sum(π) = 1
    A = np.vstack((transition_matrix.T - np.eye(n_states), np.ones((1, n_states))))
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


def simulate_markov_chain(transition_matrix, initial_state, n_steps):
    """
    Simulate a Markov chain given a transition matrix and an initial state.

    Parameters
    ----------
    transition_matrix : np.ndarray
        The transition matrix of the Markov chain.
    initial_state : int
        The initial state of the Markov chain.
    n_steps : int
        The number of steps to simulate.

    Returns
    -------
    state_trajectory : list
        A list of states visited during the simulation.
    """
    state_trajectory = [initial_state]
    current_state = initial_state
    states = np.arange(transition_matrix.shape[0])

    for _ in range(n_steps):
        current_state = np.random.choice(
            states,
            p=transition_matrix[current_state]
        )
        state_trajectory.append(current_state)

    return np.array(state_trajectory)
