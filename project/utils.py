from typing import Callable, List, Literal
import numpy as np
import logging
import scipy.stats as ss
from dataclasses import dataclass
from functools import lru_cache
import pandas as pd
from abc import ABC, abstractmethod
from scipy.optimize import minimize
import json
import os
import sympy
from sympy.utilities.lambdify import lambdify
from pathlib import Path


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

    # def update_distribution(self, time: float, initial_state_distribution: np.ndarray) -> np.ndarray:
    #     """
    #     Update the state distribution based on the transition matrix.
    #     """
    #     return state_distribution @ self.transition_matrix


def total_variation(u, v):
    return .5*np.sum(np.abs(u-v))


@lru_cache(maxsize=10)
def get_cached_queue_model(producer_lambda: float,
                           consumer_lambda: float,
                           size: int,
                           time_step: float) -> TimeContinuousMarkovChainQueueModel:
    return TimeContinuousMarkovChainQueueModel(
        producer_lambda=producer_lambda,
        consumer_lambda=consumer_lambda,
        size=size,
        time_step=time_step
    )


class Estimator(ABC):

    @abstractmethod
    def __call__(self, time: float, producer_lambda: float, queue_size_distribution: np.ndarray) -> float:
        """
        Estimate the value based on the current time, producer lambda and queue size distribution.
        """
        return NotImplementedError("This method should be implemented by subclasses.")


class PassThroughEstimator(Estimator):
    def __init__(self):
        pass

    def __call__(self, time: float,
                 producer_lambda: float,
                 queue_size_distribution: np.ndarray) -> float:
        return producer_lambda


class Controller(ABC):
    @abstractmethod
    def __call__(self, time: float,
                 producer_lambda: float,
                 queue_size_distribution: np.ndarray) -> float:
        """
        Compute the control action based on the current time and input value.
        """
        return NotImplementedError("This method should be implemented by subclasses.")


class ProportionalGainController(Controller):

    def __init__(self,
                 gain: float):
        self.gain = gain

    def __call__(self, time: float,
                 producer_lambda: float,
                 queue_size_distribution: np.ndarray) -> float:
        return self.gain * producer_lambda


class Simulation:
    def __init__(self,
                 initial_queue_size_distribution: np.ndarray,
                 queue_size: int,
                 time_step: float,
                 producer_lambda_function: Callable[[float], float],
                 producer_lambda_estimator: Estimator,
                 controller: Controller,
                 infra_cost: float,
                 wait_time_cost: float,
                 customer_revenue: float,
                 cache_models: bool = False):
        self.initial_queue_size_distribution = initial_queue_size_distribution
        self.queue_size = queue_size
        self.time_step = time_step
        self.producer_lambda_function = producer_lambda_function
        self.producer_lambda_estimator = producer_lambda_estimator
        self.controller = controller
        self.infra_cost = infra_cost
        self.wait_time_cost = wait_time_cost
        self.customer_revenue = customer_revenue
        self.cache_models = cache_models
        if cache_models:
            self.get_queue_model = get_cached_queue_model
        else:
            self.get_queue_model = TimeContinuousMarkovChainQueueModel

        # Records
        self.time: List[float] = []
        self.queue_size_distribuition: List[int] = []
        self.producer_lambda: List[float] = []
        self.producer_lambda_hat: List[float] = []
        self.effective_producer_lambda: List[float] = []
        self.consumer_lambda: List[float] = []
        self.profit: List[float] = []
        self.cost: List[float] = []
        self.revenue: List[float] = []
        self.reject_prob: List[float] = []
        self.integrated_profit: List[float] = []
        self.integrated_revenue: List[float] = []
        self.integrated_cost: List[float] = []
        self.expected_queue_size: List[float] = []
        self.expected_wait_time: List[float] = []

    def run(self, seconds: float):
        current_time = 0
        current_integrated_profit = 0
        current_integrated_revenue = 0
        current_integrated_cost = 0
        current_queue_size_distribution = self.initial_queue_size_distribution

        n_steps = int(seconds / self.time_step)
        for current_step in range(n_steps):
            current_time = current_step * self.time_step
            logging.debug(f'{current_step} - Running time {current_time}')
            self.time.append(current_time)
            self.queue_size_distribuition.append(
                current_queue_size_distribution)
            current_reject_prob = current_queue_size_distribution[-1]
            self.reject_prob.append(current_reject_prob)

            current_expected_queue_size = np.dot(
                np.arange(len(current_queue_size_distribution)),
                current_queue_size_distribution
            )
            self.expected_queue_size.append(current_expected_queue_size)

            current_producer_lambda = self.producer_lambda_function(
                current_time)
            logging.debug(f'Current producer lambda: {current_producer_lambda}')
            self.producer_lambda.append(current_producer_lambda)
            current_effective_producer_lambda = current_producer_lambda * (
                1 - current_reject_prob)
            self.effective_producer_lambda.append(
                current_effective_producer_lambda)
            current_expected_wait_time = current_expected_queue_size / \
                current_effective_producer_lambda
            self.expected_wait_time.append(current_expected_wait_time)

            current_producer_lambda_hat = self.producer_lambda_estimator(
                current_time, current_producer_lambda, current_queue_size_distribution)
            self.producer_lambda_hat.append(current_producer_lambda_hat)

            current_consumer_lambda = self.controller(
                current_time, current_producer_lambda_hat, current_queue_size_distribution)
            self.consumer_lambda.append(current_consumer_lambda)

            current_profit, current_revenue, current_cost = self.profit_function(
                consumer_lambda=current_consumer_lambda,
                effective_producer_lambda=current_effective_producer_lambda,
                expected_wait_time=current_expected_wait_time
            )
            self.profit.append(current_profit)
            self.revenue.append(current_revenue)
            self.cost.append(current_cost)

            current_integrated_profit += current_profit * self.time_step
            current_integrated_revenue += current_revenue * self.time_step
            current_integrated_cost += current_cost * self.time_step
            self.integrated_profit.append(current_integrated_profit)
            self.integrated_revenue.append(current_integrated_revenue)
            self.integrated_cost.append(current_integrated_cost)

            queue_model = self.get_queue_model(
                producer_lambda=current_producer_lambda,
                consumer_lambda=current_consumer_lambda,
                size=self.queue_size,
                time_step=self.time_step
            )
            current_queue_size_distribution = queue_model.discrete.update_distribution(
                current_queue_size_distribution)

        return self.integrated_profit[-1]

    def profit_function(self,
                        consumer_lambda: float,
                        effective_producer_lambda: float,
                        expected_wait_time: float) -> float:

        revenue = self.customer_revenue * effective_producer_lambda
        cost = self.infra_cost * consumer_lambda + \
            self.wait_time_cost * expected_wait_time
        profit = revenue - cost
        return profit, revenue, cost

    def dump_results(self, filename: str):
        results_df = dict(
            time=self.time,
            queue_size_distribuition=self.queue_size_distribuition,
            producer_lambda=self.producer_lambda,
            producer_lambda_hat=self.producer_lambda_hat,
            effective_producer_lambda=self.effective_producer_lambda,
            consumer_lambda=self.consumer_lambda,
            profit=self.profit,
            cost=self.cost,
            revenue=self.revenue,
            reject_prob=self.reject_prob,
            integrated_profit=self.integrated_profit,
            integrated_revenue=self.integrated_revenue,
            integrated_cost=self.integrated_cost,
            expected_queue_size=self.expected_queue_size,
            expected_wait_time=self.expected_wait_time,
        )
        results_df = pd.DataFrame.from_dict(results_df)
        results_df.to_parquet(filename)


class Optimizer:

    def __init__(self,
                 initial_queue_size_distribution: np.ndarray,
                 initial_controller_gain: float,
                 producer_lambda_function: str | Callable[[float], float],
                 queue_size: int,
                 infra_cost: float,
                 wait_time_cost: float,
                 customer_revenue: float,
                 seconds: float,
                 time_step: float,
                 output_dir: str,
                 cache_models: bool = False):
        self.initial_queue_size_distribution = initial_queue_size_distribution
        self.initial_controller_gain = initial_controller_gain
        if isinstance(producer_lambda_function, str):
            self.producer_lambda_function = producer_lambda_function
            sympy_locals = dict(t=sympy.Symbol("t", real=True))
            sym_func = sympy.sympify(producer_lambda_function, locals=sympy_locals)
            self.producer_lambda_callable = lambdify(
                "t", sym_func, modules="numpy")
        else:
            self.producer_lambda_function = "Not a string"
            self.producer_lambda_callable = producer_lambda_function
        self.queue_size = queue_size
        self.infra_cost = infra_cost
        self.wait_time_cost = wait_time_cost
        self.customer_revenue = customer_revenue
        self.seconds = seconds
        self.time_step = time_step
        self.output_dir = output_dir
        self.cache_models = cache_models

        self.controller_gain: List[float] = []
        self.costs: List[float] = []
        self.call_counter = 0
        self.result = None

    def cost_function(self, x: np.ndarray) -> float:
        controller_gain = x[0]
        logging.info(
            f'Optimizer call {self.call_counter} with controller_gain={controller_gain}')
        self.controller_gain.append(float(controller_gain))
        controller = ProportionalGainController(gain=controller_gain)
        sim = Simulation(
            initial_queue_size_distribution=self.initial_queue_size_distribution,
            queue_size=self.queue_size,
            time_step=self.time_step,
            producer_lambda_function=self.producer_lambda_callable,
            producer_lambda_estimator=PassThroughEstimator(),
            controller=controller,
            infra_cost=self.infra_cost,
            wait_time_cost=self.wait_time_cost,
            customer_revenue=self.customer_revenue,
            cache_models=self.cache_models
        )
        cost = -sim.run(self.seconds)
        self.costs.append(float(cost))
        self.call_counter += 1
        if self.output_dir:
            sim.dump_results(
                f'{self.output_dir}/simulation_{self.call_counter:03}.parquet')
        return cost

    def run(self):
        numpy_result = minimize(
            self.cost_function,
            x0=np.array([self.initial_controller_gain]),
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-8}
        )
        self._set_numpy_result(numpy_result)
        return self.result

    def _set_numpy_result(self, result):
        self.result = dict(
            x=float(result.x[0]),
            fun=float(result.fun),
            status=result.status,
            success=result.success,
            message=result.message,
            nfev=result.nfev,
            nit=result.nit
        )

    def as_dict(self):
        return dict(
            controller_gain=self.controller_gain,
            costs=self.costs,
            initial_queue_size_distribution=self.initial_queue_size_distribution.tolist(),
            initial_controller_gain=float(self.initial_controller_gain),
            producer_lambda_function=self.producer_lambda_function,
            queue_size=float(self.queue_size),
            infra_cost=float(self.infra_cost),
            wait_time_cost=float(self.wait_time_cost),
            customer_revenue=float(self.customer_revenue),
            seconds=float(self.seconds),
            time_step=float(self.time_step),
            call_counter=self.call_counter,
            result=self.result
        )

    def dump_results(self):
        if not self.output_dir:
            raise ValueError(
                "Output directory is not set. Cannot dump results.")
        filename = os.path.join(self.output_dir, 'optimization_results.json')
        with open(filename, 'w') as f:
            json.dump(self.as_dict(), f, indent=4)

    @classmethod
    def from_dir(cls, output_dir: str | Path) -> 'Optimizer':
        if isinstance(output_dir, str):
            with open(os.path.join(output_dir, 'optimization_results.json'), 'r') as f:
                data = json.load(f)
        elif isinstance(output_dir, Path):
            with open(output_dir / 'optimization_results.json', 'r') as f:
                data = json.load(f)
        else:
            raise ValueError("output_dir must be a string or Path object.")
        instance = cls(
            initial_queue_size_distribution=np.array(data['initial_queue_size_distribution']),
            initial_controller_gain=data['initial_controller_gain'],
            producer_lambda_function=data['producer_lambda_function'],
            queue_size=data['queue_size'],
            infra_cost=data['infra_cost'],
            wait_time_cost=data['wait_time_cost'],
            customer_revenue=data['customer_revenue'],
            seconds=data['seconds'],
            time_step=data['time_step'],
            output_dir=output_dir
        )
        instance.controller_gain = np.array(data['controller_gain'])
        instance.costs = np.array(data['costs'])
        instance.call_counter = data['call_counter']
        instance.result = data['result']
        return instance

    def get_sim(self, iteration: int | Literal['best', 'worst']) -> pd.DataFrame:
        """
        Get the simulation results for a specific iteration.
        """
        if not self.output_dir:
            raise ValueError("Output directory is not set. Cannot get simulation results.")
        if iteration == 'best':
            iteration = np.argmin(self.costs)
        elif iteration == 'worst':
            iteration = np.argmax(self.costs)
        filename = os.path.join(self.output_dir, f'simulation_{iteration:03}.parquet')
        return pd.read_parquet(filename)
