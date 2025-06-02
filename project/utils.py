from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.stats as ss
from dataclasses import dataclass
from functools import cache, partial
from scipy.optimize import minimize
import pandas as pd


def heatmap(data, row_labels=None, col_labels=None, ax=None,
            cbar_kw=None, cbarlabel="", fontsize='medium', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

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
class LSTSQReport:
    stationary_distribution: np.ndarray
    reject_prob: float
    residuals: np.ndarray
    rank: int
    singular_values: np.ndarray
    expected_stationary_state: float
    expected_wait_time: float
    effective_producer_lambda: float

    @classmethod
    def from_transition_matrix(cls,
                               transition_matrix: np.ndarray,
                               producer_lambda: float) -> 'LSTSQReport':
        n_states = len(transition_matrix)
        A = np.vstack(
            (transition_matrix - np.eye(n_states), np.ones((1, n_states))))
        b = np.zeros(n_states + 1)
        b[-1] = 1
        res = np.linalg.lstsq(A, b, rcond=0)
        stationary_distribution = res[0].flatten()
        expected_stationary_state = np.dot(
            np.arange(len(stationary_distribution)), stationary_distribution
        )
        reject_prob = stationary_distribution[-1]
        effective_producer_lambda = producer_lambda * (1 - reject_prob)
        expected_wait_time = expected_stationary_state / effective_producer_lambda
        return cls(
            stationary_distribution=stationary_distribution,
            reject_prob=stationary_distribution[-1],
            residuals=res[1],
            rank=res[2],
            singular_values=res[3],
            expected_stationary_state=expected_stationary_state,
            expected_wait_time=expected_wait_time,
            effective_producer_lambda=effective_producer_lambda
        )


@dataclass(frozen=True)
class EigenvaluesReport:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_gap: float
    stationary_distribution: np.ndarray
    reject_prob: float
    expected_stationary_state: float
    expected_wait_time: float
    effective_producer_lambda: float

    @classmethod
    def from_values_vectors(cls,
                            eigenvalues: np.ndarray,
                            eigenvectors: np.ndarray,
                            producer_lambda: float) -> 'EigenvaluesReport':
        sortidx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sortidx]
        eigenvectors = eigenvectors[:, sortidx]
        spectral_gap = 1 - np.abs(eigenvalues[1])
        stationary_distribution = eigenvectors[:, 0]
        if np.all(stationary_distribution < 0):
            stationary_distribution = -stationary_distribution
        stationary_distribution /= np.sum(stationary_distribution**2)
        expected_stationary_state = np.dot(
            np.arange(len(stationary_distribution)), stationary_distribution
        )
        reject_prob = stationary_distribution[-1]
        effective_producer_lambda = producer_lambda * (1 - reject_prob)
        expected_wait_time = expected_stationary_state / effective_producer_lambda

        return cls(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_gap=spectral_gap,
            stationary_distribution=stationary_distribution,
            reject_prob=reject_prob,
            expected_stationary_state=expected_stationary_state,
            expected_wait_time=expected_wait_time,
            effective_producer_lambda=effective_producer_lambda
        )


class MarkovChainQueueModel:

    @property
    def eigenvalues_report(self) -> EigenvaluesReport:
        """
        Compute the eigenvalues and eigenvectors of the transition matrix.
        The stationary distribution is computed as the normalized eigenvector
        corresponding to the eigenvalue 1.
        """
        if not hasattr(self, '_eigenvalues_report'):
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            self._eigenvalues_report = EigenvaluesReport.from_values_vectors(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                producer_lambda=self.producer_lambda
            )
        return self._eigenvalues_report

    @property
    def lstsq_report(self) -> LSTSQReport:
        if not hasattr(self, '_lstsq_report'):
            self._lstsq_report = LSTSQReport.from_transition_matrix(
                self.transition_matrix.T,
                self.producer_lambda
            )
        return self._lstsq_report


class DiscreteMarkovChainQueueModel(MarkovChainQueueModel):
    def __init__(self,
                 producer_lambda: float,
                 consumer_lambda: float,
                 size: int):
        self.producer_lambda = producer_lambda
        self.consumer_lambda = consumer_lambda
        self.lambda_rate = self.producer_lambda / self.consumer_lambda
        self.size = size
        self.produced_items = ss.poisson(self.producer_lambda)
        self.consumed_items = ss.poisson(self.consumer_lambda)
        self.liquid_items = ss.skellam(self.producer_lambda,
                                       self.consumer_lambda)
        self.transition_matrix = self._compute_transition_matrix()

    def _compute_transition_matrix(self):
        transition_matrix = np.zeros(
            (self.size + 1, self.size + 1))

        for i in range(self.size + 1):
            transition_matrix[i, 0] = self.liquid_items.cdf(-i)
            transition_matrix[i, self.size] = 1 - \
                self.liquid_items.cdf(self.size - i - 1)
            transition_matrix[i, 1:self.size] = self.liquid_items.pmf(
                np.arange(1-i, self.size-i))

        return transition_matrix

    def update_distribution(self, state_distribution: np.ndarray) -> np.ndarray:
        """
        Update the state distribution based on the transition matrix.
        """
        return state_distribution @ self.transition_matrix


class TimeContinuousMarkovChainQueueModel(MarkovChainQueueModel):

    def __init__(self,
                 producer_lambda: float,
                 consumer_lambda: float,
                 size: int,
                 time_step: float = 1.0):
        self.producer_lambda = producer_lambda
        self.consumer_lambda = consumer_lambda
        self.size = size
        self.time_step = time_step

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

        self.discrete = DiscreteMarkovChainQueueModel(
            self.time_step*self.producer_lambda,
            self.time_step*consumer_lambda,
            self.size
        )

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


class PassThroughEstimator:
    def __init__(self):
        pass

    def __call__(self, time: float, input_value: float) -> float:
        return input_value


class ProportionalGain:

    def __init__(self,
                 gain: float):
        self.gain = gain

    def __call__(self, time: float, input_value: float) -> float:
        return self.gain * input_value


class Profit:

    def __init__(self,
                 infra_cost: float,
                 wait_time_cost: float,
                 customer_revenue: float):
        self.infra_cost = infra_cost
        self.wait_time_cost = wait_time_cost
        self.customer_revenue = customer_revenue

    def __call__(self,
                 consumer_lambda: float,
                 effective_producer_lambda: float,
                 expected_wait_time: float) -> float:
        revenue = self.customer_revenue * effective_producer_lambda
        cost = self.infra_cost * consumer_lambda + \
            self.wait_time_cost * expected_wait_time
        profit = revenue - cost
        return profit, revenue, cost

    def sample_computation(
        self,
        consumer_lambda: float,
        new_clients: float,
        expected_wait_time: float
    ) -> float:
        """
        Sample computation of the profit function.
        This is a simplified version that does not take into account the
        infrastructure costs and the wait time costs.
        """
        revenue = self.customer_revenue * new_clients
        cost = self.infra_cost * consumer_lambda + \
            self.wait_time_cost * expected_wait_time
        profit = revenue - cost
        return profit, revenue, cost


class MPCGainEstimator:

    def __init__(self,
                 n_steps: int,
                 profit_function: Callable[[float, float, float], Tuple[float, float, float]],
                 controller_class,
                 queue_size: int,
                 time_step: float):
        self.n_steps = n_steps
        self.profit_function = profit_function
        self.controller_class = controller_class
        self.queue_size = queue_size
        self.time_step = time_step

    def _cost_function(self,
                       controller_params: np.ndarray,
                       producer_lambda_hat: float,
                       start_queue_size: int) -> float:
        controller = self.controller_class(gain=controller_params[0])
        queue_size_distribution = np.zeros(self.queue_size + 1)
        queue_size_distribution[start_queue_size] = 1.0
        current_queue_size_distribution = queue_size_distribution
        estimated_cost = 0
        estimated_revenue = 0
        estimated_profit = 0
        logging.debug(
            f'Starting cost function computation with controller params: {controller_params}  for {self.n_steps} steps')
        for current_step in range(self.n_steps):
            logging.debug(f'Running step {current_step}')
            current_time = current_step * self.time_step
            consumer_lambda = controller(current_time, producer_lambda_hat)
            queue_model = TimeContinuousMarkovChainQueueModel(
                producer_lambda=producer_lambda_hat,
                consumer_lambda=consumer_lambda,
                size=self.queue_size,
                time_step=self.time_step
            )
            reject_prob = current_queue_size_distribution[-1]
            effective_producer_lambda_hat = producer_lambda_hat * \
                (1 - reject_prob)
            expected_queue_size = np.dot(
                np.arange(self.queue_size + 1),
                current_queue_size_distribution
            )
            expected_wait_time = expected_queue_size / effective_producer_lambda_hat
            profit, revenue, cost = self.profit_function(
                consumer_lambda, effective_producer_lambda_hat, expected_wait_time
            )
            estimated_profit += profit * self.time_step
            estimated_revenue += revenue * self.time_step
            estimated_cost += cost * self.time_step
            current_queue_size_distribution =\
                queue_model.discrete.update_distribution(
                    current_queue_size_distribution)

        return -estimated_profit

    def __call__(self, time: float,
                 producer_lambda_hat: float,
                 current_gain: float,
                 current_queue_size: int) -> float:

        cost_function = partial(
            self._cost_function,
            producer_lambda_hat=producer_lambda_hat,
            start_queue_size=current_queue_size
        )
        logging.debug(f'Runing optimization for time {time} with initial gain {current_gain}')
        res = minimize(
            cost_function,
            x0=current_gain,
            method='Nelder-Mead',
            options={'xatol': 1e-8}
        )
        if not res.success:
            logging.warning(
                f"Optimization failed at time {time}: {res.message}")
            return current_gain

        controller = self.controller_class(res.x)
        consumer_lambda = controller(time, producer_lambda_hat)

        return consumer_lambda, res.x, -res.func


# class Optimizer:

#     def __init__(self,
#                  seconds: float,
#                  profit_function: Callable[[float, float, float], Tuple[float, float, float]],
#                  controller_class,
#                  queue_size: int,
#                  time_step: float,
#                  producer_lambda_function: Callable[[float], float],
#                  producer_lambda_estimator: Callable[[float, float], float]):
#         self.n_steps = seconds
#         self.profit_function = profit_function
#         self.controller_class = controller_class
#         self.queue_size = queue_size
#         self.time_step = time_step
#         self.n_steps = int(seconds / time_step)
#         self.producer_lambda_function = producer_lambda_function
#         self.producer_lambda_estimator = producer_lambda_estimator
        

#     def cost_function(self,
#                       controller_params: np.ndarray,
#                       producer_lambda_estimator: float,
#                       start_queue_size: int) -> float:
#         controller = self.controller_class(gain=controller_params[0])
#         queue_size_distribution = np.zeros(self.queue_size + 1)
#         queue_size_distribution[start_queue_size] = 1.0
#         current_queue_size_distribution = queue_size_distribution
#         estimated_cost = 0
#         estimated_revenue = 0
#         estimated_profit = 0
#         logging.debug(
#             f'Starting cost function computation with controller params: {controller_params}  for {self.n_steps} steps')
#         for current_step in range(self.n_steps):
#             logging.debug(f'Running step {current_step}')
#             current_time = current_step * self.time_step
#             cur
#             current_producer_lambda_hat = self.producer_lambda_estimator(time, )
#             consumer_lambda = controller(current_time, producer_lambda_hat)
#             queue_model = TimeContinuousMarkovChainQueueModel(
#                 producer_lambda=producer_lambda_hat,
#                 consumer_lambda=consumer_lambda,
#                 size=self.queue_size,
#                 time_step=self.time_step
#             )
#             reject_prob = current_queue_size_distribution[-1]
#             effective_producer_lambda_hat = producer_lambda_hat * \
#                 (1 - reject_prob)
#             expected_queue_size = np.dot(
#                 np.arange(self.queue_size + 1),
#                 current_queue_size_distribution
#             )
#             expected_wait_time = expected_queue_size / effective_producer_lambda_hat
#             profit, revenue, cost = self.profit_function(
#                 consumer_lambda, effective_producer_lambda_hat, expected_wait_time
#             )
#             estimated_profit += profit * self.time_step
#             estimated_revenue += revenue * self.time_step
#             estimated_cost += cost * self.time_step
#             current_queue_size_distribution =\
#                 queue_model.discrete.update_distribution(
#                     current_queue_size_distribution)

#         return -estimated_profit

#     def minimize(self, time: float,
#                  producer_lambda_hat: float,
#                  current_gain: float,
#                  current_queue_size: int) -> float:

#         cost_function = partial(
#             self._cost_function,
#             producer_lambda_hat=producer_lambda_hat,
#             start_queue_size=current_queue_size
#         )
#         logging.debug(f'Runing optimization for time {time} with initial gain {current_gain}')
#         res = minimize(
#             cost_function,
#             x0=current_gain,
#             method='Nelder-Mead',
#             options={'xatol': 1e-8}
#         )
#         if not res.success:
#             logging.warning(
#                 f"Optimization failed at time {time}: {res.message}")
#             return current_gain

#         controller = self.controller_class(res.x)
#         consumer_lambda = controller(time, producer_lambda_hat)

#         return consumer_lambda, res.x, -res.func


class Simulation:
    def __init__(self,
                 initial_queue_size: int,
                 queue_size: int,
                 time_step: float,
                 producer_lambda_function: Callable[[float], float],
                 controller_window: float,
                 infra_cost: float,
                 wait_time_cost: float,
                 customer_revenue: float,
                 start_controller_gain: float = 1.0):
        self.initial_queue_size = initial_queue_size
        self.queue_size = queue_size
        self.time_step = time_step
        self.producer_lambda_function = producer_lambda_function
        self.infra_cost = infra_cost
        self.wait_time_cost = wait_time_cost
        self.customer_revenue = customer_revenue
        self.start_controller_gain = start_controller_gain
        self.profit_function = Profit(
            infra_cost=infra_cost,
            wait_time_cost=wait_time_cost,
            customer_revenue=customer_revenue
        )
        self.controller_steps = int(controller_window / time_step)
        self.controller = MPCGainEstimator(
            n_steps=self.controller_steps,
            profit_function=self.profit_function,
            controller_class=ProportionalGain,
            queue_size=queue_size,
            time_step=self.time_step
        )

        # Records
        self.time: List[float] = []
        self.queue_size: List[int] = []
        self.producer_lambda: List[float] = []
        self.producer_lambda_hat: List[float] = []
        self.consumer_lambda: List[float] = []
        self.expected_wait_time: List[float] = []
        self.controller_gain: List[float] = []
        self.profit: List[float] = []

    def run(self, seconds: float):
        current_step = 0
        current_time = 0
        current_queue_size = self.initial_queue_size
        current_controller_gain = self.start_controller_gain

        n_steps = int(seconds / self.time_step)
        for current_step in range(n_steps):
            current_time = current_step * self.time_step
            logging.debug(f'{current_step} - Running time {current_time}')
            self.time.append(current_time)

            self.queue_size.append(current_queue_size)

            current_producer_lambda = self.producer_lambda_function(
                current_time)
            self.producer_lambda.append(current_producer_lambda)
            current_producer_lambda_hat = current_producer_lambda
            self.producer_lambda_hat.append(current_producer_lambda_hat)

            current_consumer_lambda, controller_params, profit = self.controller(
                time=current_time,
                producer_lambda_hat=current_producer_lambda_hat,
                current_gain=current_controller_gain,
                current_queue_size=current_queue_size)
            current_controller_gain = controller_params[0]
            self.consumer_lambda.append(current_consumer_lambda)
            self.profit.append(profit)
            expected_wait_time = current_queue_size / current_consumer_lambda
            self.expected_wait_time.append(expected_wait_time)
            self.controller_gain.append(current_controller_gain)

            # new_clients = ss.poisson.rvs(
            #     self.time_step * current_producer_lambda_hat, 1
            # )
            # processed_clients = ss.poisson.rvs(
            #     self.time_step * current_consumer_lambda, 1
            # )

            # current_profit, current_revenue, current_cost = \
            #     self.profit_function.sample_computation(
            #         consumer_lambda=current_consumer_lambda,
            #         effective_producer_lambda=current_producer_lambda_hat,
            #         expected_wait_time=expected_wait_time
            #     )

            queue_model = get_cached_queue_model(
                producer_lambda=current_producer_lambda,
                consumer_lambda=current_consumer_lambda,
                queue_size=self.queue_size
            )

            current_queue_size = queue_model.update_state(current_queue_size)

    def dump_results(self, filename: str):
        results_df = dict(
            time=self.time,
            queue_size=self.queue_size,
            producer_lambda=self.producer_lambda,
            producer_lambda_hat=self.producer_lambda_hat,
            consumer_lambda=self.consumer_lambda,
            expected_wait_time=self.expected_wait_time,
            controller_gain=self.controller_gain,
            profit=self.profit
        )
        results_df = pd.DataFrame.from_dict(results_df)
        results_df.to_parquet(filename)
