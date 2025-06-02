# Source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

from typing import Callable, Literal
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.stats as ss
from dataclasses import dataclass
from functools import cache


def heatmap(data, row_labels=None, col_labels=None, ax=None,
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
    if col_labels is not None:
        ax.set_xticks(range(data.shape[1]), labels=col_labels,
                      size=fontsize,
                      rotation=-30, ha="right", rotation_mode="anchor")
    if row_labels is not None:
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


@dataclass(frozen=True)
class LSTSQStationaryDistributionReport:
    distribution: np.ndarray
    residuals: np.ndarray
    rank: int
    singular_values: np.ndarray


class DiscreteMarkovChain:
    def __init__(self,
                 transition_function: Callable[[int], int]):
        self.transition_function = transition_function

    def update_state(self, current_state: int) -> int:
        self.current_state = self.transition_function(current_state)


class DiscreteFiniteMarkovChain(DiscreteMarkovChain):
    def __init__(self,
                 transition_matrix: np.ndarray):
        self.transition_matrix = transition_matrix
        self.n_states = transition_matrix.shape[0]

        super().__init__(self.transition_function)

    def __repr__(self):
        return f"KnownFiniteMarkovChain(n_states={self.n_states})"

    def transition_function(self, current_state: int) -> int:
        """
        Transition function for the Markov chain.
        Returns the next state based on the current state.
        """
        if isinstance(current_state, str):
            current_state = self.states_labels[current_state]
        return np.random.choice(
            self.n_states,
            p=self.transition_matrix[current_state]
        )

    def update_distribution(self, current_state_distribution: np.ndarray) -> np.ndarray:
        """
        Update the state distribution based on the current state distribution
        and the transition matrix.
        """
        return current_state_distribution @ self.transition_matrix

    @property
    def lstsq_distribution(self) -> np.ndarray:
        if not hasattr(self, '_lstsq_distribution'):
            self._compute_lstsq_statationary_distribution()
        return self._lstsq_distribution

    @property
    def lstsq_report(self) -> LSTSQStationaryDistributionReport:
        if not hasattr(self, '_lstsq_report'):
            self._compute_lstsq_statationary_distribution()
        return self._lstsq_report

    @property
    def expected_stationary_state(self):
        if not hasattr(self, '_expected_stationary_state'):
            self._compute_lstsq_statationary_distribution()
        return self._expected_stationary_state

    def _compute_lstsq_statationary_distribution(self):
        # Solve the equation πP = π
        # where sum(π) = 1
        A = np.vstack((self.transition_matrix.T -
                      np.eye(self.n_states), np.ones((1, self.n_states))))
        b = np.zeros(self.n_states + 1)
        b[-1] = 1
        res = np.linalg.lstsq(A, b, rcond=0)
        self._lstsq_distribution = res[0].flatten()
        self._lstsq_report = LSTSQStationaryDistributionReport(
            distribution=self._lstsq_distribution,
            residuals=res[1],
            rank=res[2],
            singular_values=res[3],
        )
        self._expected_stationary_state = np.dot(
            np.arange(self.n_states),
            self._lstsq_distribution
        )

    @property
    def spectral_gap(self):
        if not hasattr(self, '_gap'):
            self._compute_spectral_gap()
        return self._spectral_gap

    @property
    def eigenvalues(self):
        if not hasattr(self, '_eigenvalues'):
            self._compute_spectral_gap()
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if not hasattr(self, '_eigenvectors'):
            self._compute_spectral_gap()
        return self._eigenvectors

    @property
    def eigenvector_stationary(self):
        """
        Compute the stationary distribution using the eigenvector method.
        The stationary distribution is the normalized eigenvector corresponding to the
        eigenvalue 1 of the transition matrix.
        """
        if not hasattr(self, '_eigenvector_stationary'):
            self._compute_spectral_gap()
        return self._eigenvector_stationary

    def _compute_spectral_gap(self):
        # Get the eigenvalues of the Markov matrix
        self._eigenvalues, self._eigenvectors = np.linalg.eig(
            self.transition_matrix)
        sortidx = np.argsort(np.abs(self._eigenvalues))[::-1]
        self._eigenvalues = self._eigenvalues[sortidx]
        self._eigenvectors = self._eigenvectors[:, sortidx]
        abs_eigenvalues = np.abs(self._eigenvalues)
        self._spectral_gap = 1 - abs_eigenvalues[1]
        self._eigenvector_stationary = self._eigenvectors[:, 0]
        self._eigenvector_stationary /= np.sum(self._eigenvector_stationary**2)

    def compute_mixture_time(self, epsilon, n_steps=10000,
                             return_type: Literal['time', 'max_path', 'all_paths'] = 'time'):
        logging.info('Computing the mixture time')
        current_state_distribution = np.eye(self.n_states)
        transition_counter = 0
        stationary_matrix = [
            self.stationary_distribution for _ in range(self.n_states)
        ]
        stationary_matrix = np.array(stationary_matrix)
        current_variations = 0.5*np.sum(
            np.abs(current_state_distribution - stationary_matrix),
            axis=1
        )
        return_is_max = return_type == 'max_path'
        return_is_all = return_type == 'all_paths'
        if return_is_max:
            variation_path = [max(current_variations)]
        elif return_is_all:
            variation_path = [current_variations]

        while np.all(current_variations > epsilon) and (transition_counter < n_steps):
            # Compute the variation distance
            # between the current state and the stationary distribution
            transition_counter += 1
            current_state_distribution = current_state_distribution @ self.transition_matrix
            current_variations = 0.5*np.sum(
                np.abs(current_state_distribution - stationary_matrix),
                axis=1
            )
            if return_is_max:
                variation_path.append(max(current_variations))
            elif return_is_all:
                variation_path.append(current_variations)

        if transition_counter >= n_steps:
            logging.warning("The Markov chain misture time did not converge"
                            f" within {n_steps} iterations and epsilon={epsilon}.")
        else:
            logging.info(f"The Markov chain mixture time converged in "
                         f"{transition_counter} iterations.")
            return transition_counter, np.array(-1)

        if return_type == 'time':
            return transition_counter, None
        else:
            return transition_counter, np.array(variation_path)


class TimeContinuousMarkovChainQueueModel(DiscreteMarkovChain):

    def __init__(self,
                 producer_lambda: float,
                 consumer_lambda: float,
                 size: int,
                 time_step: float = 1.0):
        self.producer_lambda = producer_lambda
        self.consumer_lambda = consumer_lambda
        self.size = size
        self.time_step = time_step

        self.produced_items = ss.poisson(self.time_step*self.producer_lambda)
        self.consumed_items = ss.poisson(self.time_step*self.consumer_lambda)
        self.liquid_items = ss.skellam(self.time_step*self.producer_lambda,
                                       self.time_step*self.consumer_lambda)
        self.producer_time = ss.expon(scale=1/self.producer_lambda)
        self.consumer_time = ss.expon(scale=1/self.consumer_lambda)

        lambda_sum = self.producer_lambda + self.consumer_lambda
        self.increase_prob = self.producer_lambda / lambda_sum
        self.decrease_prob = self.consumer_lambda / lambda_sum
        self.lambda_rate = self.producer_lambda / self.consumer_lambda

        self.stationary_distribution = self._compute_stationary_distribution()
        self.expected_size = np.dot(
            np.arange(self.size + 1),
            self.stationary_distribution
        )
        self.effective_producer_lambda = self.producer_lambda*(
            1 - self.stationary_distribution[-1])
        self.expected_wait_time = self.expected_size / self.effective_producer_lambda
        self.transition_matrix = self._compute_transition_matrix()

        self.discrete = DiscreteFiniteMarkovChain(
            self._get_discrete_transition_matrix()
        )
        super().__init__(self.discrete.transition_function)

    def _compute_stationary_distribution(self):
        """
        Compute the continuous stationary distribution of the Markov chain.
        The continuous stationary distribution is a vector of size (size + 1)
        where each element corresponds to the probability of being in that state.
        """
        if self.lambda_rate == 1:
            pi_zero = 1 / (self.size + 1)
        else:
            pi_zero = (1-self.lambda_rate) / \
                (1 - (self.lambda_rate**(self.size + 1)))

        return pi_zero * np.power(self.lambda_rate, np.arange(self.size + 1))

    def _compute_transition_matrix(self):
        transition_matrix = np.zeros(
            (self.size + 1, self.size + 1))

        for i in range(self.size + 1):
            # Probability of increasing the queue size
            if i < self.size:
                transition_matrix[i, i + 1] = self.increase_prob
            else:
                transition_matrix[i, i] = self.increase_prob

            # Probability of decreasing the queue size
            if i > 0:
                transition_matrix[i, i - 1] = self.decrease_prob
            else:
                transition_matrix[i, i] = self.decrease_prob

        return transition_matrix

    def _get_discrete_transition_matrix(self):
        """
        Create the transition matrix for the Markov chain.
        The transition matrix is a square matrix of size (size + 1)
        where each row corresponds to a state and each column corresponds
        to a possible next state.
        """
        transition_matrix = np.zeros(
            (self.size + 1, self.size + 1))

        for i in range(self.size + 1):
            transition_matrix[i, 0] = self.liquid_items.cdf(-i)
            transition_matrix[i, self.size] = 1 - \
                self.liquid_items.cdf(self.size - i - 1)
            transition_matrix[i, 1:self.size] = self.liquid_items.pmf(
                np.arange(1-i, self.size-i))

        return transition_matrix


def total_variation(u, v):
    return .5*np.sum(np.abs(u-v))


def operation_profit(
    infra_costs: float,
    reject_prob: float,
    wait_time: float,
    producer_lambda: float,
    customer_revenue: float,
    wait_time_cost: float,
):
    revenue = customer_revenue*producer_lambda*(1 - reject_prob)
    costs = infra_costs + wait_time_cost*wait_time
    return revenue - costs


def infra_lambda(
    machines: np.ndarray,
    machines_lambda: np.ndarray
) -> float:
    """
    Calculate the total lambda of the infrastructure based on the number of machines
    and their individual lambda values.
    """
    return np.dot(machines, machines_lambda)


def infra_cost(
    machines: np.ndarray,
    operation_cost: np.ndarray
) -> float:
    """
    Calculate the total cost of the infrastructure based on the number of machines
    and their individual operation costs.
    """
    return np.dot(machines, operation_cost)


@cache
def get_cached_queue_model(producer_lambda: float,
                           consumer_lambda: float,
                           queue_size: int,
                           discrete_time_step: float = 0.01) -> TimeContinuousMarkovChainQueueModel:
    return TimeContinuousMarkovChainQueueModel(
        producer_lambda=producer_lambda,
        consumer_lambda=consumer_lambda,
        size=queue_size,
        time_step=discrete_time_step
    )
