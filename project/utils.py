
from typing import Any, Callable, Dict, List
import numpy as np
from numpy.typing import ArrayLike, NDArray
import logging
from scipy import linalg, integrate, optimize, stats
from functools import cached_property
import pandas as pd
from abc import ABC, abstractmethod
import json
import os
import sympy
from sympy.utilities.lambdify import lambdify
from pathlib import Path
from sklearn.linear_model import LinearRegression


def exponential_regression(x: NDArray[np.floating],
                           y: NDArray[np.floating],
                           sample_weight: NDArray[np.floating] | None = None
                           ) -> tuple[float, float, LinearRegression]:
    """
    Returns the parameters of an exponential regression model fitted to the data.
    The model is of the form y = A * exp(b * x), where A is the intercept and b is the slope
    of the equivalent linear model log(y) = A + b * x.

    Parameters
    ----------
    x : NDArray[np.floating]
        x values of the data points.
    y : NDArray[np.floating]
        y values of the data points.

    Returns
    -------
    tuple[float, float, LinearRegression]
        A tuple containing the intercept (A), slope (b) of the linearized model,
        and the fitted LinearRegression model.
    """
    y = np.log(y)
    model = LinearRegression()
    model.fit(x, y, sample_weight)
    A = model.intercept_
    b = model.coef_[0]
    return A, b, model


class PoissonProcess:
    def __init__(self, rate: float, size: int):
        """
        Initialize the Poisson process with the given rate.
        """
        self.rate = rate
        self.size = size
        self.inter_arrival_time_distribution = stats.expon(scale=1/rate)

    @cached_property
    def rate_matrix(self) -> NDArray[np.floating]:
        rate_matrix = np.zeros(
            (self.size + 1, self.size + 1))  # States from 0 to size
        states = np.arange(self.size + 1)  # States from 0 to size
        rate_matrix[states[:-1], states[:-1]] = -self.rate

        rate_matrix[states[:-1], states[1:]] = self.rate
        return rate_matrix


class LimitedSizeMarkovChainQueueModel(ABC):

    def __init__(self,
                 arrival_rate: float,
                 service_rate: float,
                 size: int):
        """
        Initialize the Markov chain queue model with the given parameters.
        """
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.size = size

        self.arrival_time = stats.expon(scale=1/self.arrival_rate)
        self.service_time = stats.expon(scale=1/self.service_rate)

    @property
    @abstractmethod
    def rate_matrix(self) -> np.ndarray:
        """
        The rate matrix of the Markov chain.
        This is an abstract property that must be implemented by subclasses.
        """
        return NotImplementedError("This method should be implemented by subclasses.")

    @property
    @abstractmethod
    def stationary_distribution(self) -> np.ndarray:
        """
        The stationary distribution of the Markov chain.
        This is an abstract property that must be implemented by subclasses.
        """
        return NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def get_expected_size(queue_size_distribution: np.ndarray) -> float:
        """
        Compute the expected size of the queue based on the queue size distribution.
        """
        return np.dot(np.arange(len(queue_size_distribution)), queue_size_distribution)

    @cached_property
    def expected_size(self) -> float:
        """
        Compute the expected size of the queue based on the stationary distribution.
        """
        return self.__class__.get_expected_size(self.stationary_distribution)

    @staticmethod
    def get_effective_arrival_rate(arrival_rate: float, queue_size_distribution: np.ndarray) -> float:
        """
        Compute the effective arrival rate based on the queue size distribution distribution.
        The effective arrival rate is the arrival rate multiplied by the
        probability of not rejecting an item.
        """
        return arrival_rate * (1 - queue_size_distribution[-1])

    @cached_property
    def effective_arrival_rate(self) -> float:
        """
        Compute the effective arrival rate based on the stationary distribution.
        The effective arrival rate is the arrival rate multiplied by the
        probability of not rejecting an item.
        """
        return self.__class__.get_effective_arrival_rate(
            arrival_rate=self.arrival_rate,
            queue_size_distribution=self.stationary_distribution
        )

    @staticmethod
    def get_expected_wait_time(expected_size: float, effective_arrival_rate: float) -> float:
        """
        Compute the expected wait time based on the expected size and effective arrival rate.
        The expected wait time is the expected size divided by the effective arrival rate.
        """
        if effective_arrival_rate == 0:
            return np.inf
        return expected_size / effective_arrival_rate

    @cached_property
    def expected_wait_time(self) -> float:
        """
        Compute the expected wait time based on the expected size and effective arrival rate.
        The expected wait time is the expected size divided by the effective arrival rate.
        """
        return self.__class__.get_expected_wait_time(
            expected_size=self.expected_size,
            effective_arrival_rate=self.effective_arrival_rate
        )

    def distribution(self, time: float, initial_state_distribution: np.ndarray) -> np.ndarray:
        """
        Update the state distribution based on the transition matrix.
        """
        return initial_state_distribution @ linalg.expm(self.rate_matrix * time)

    def diff_distribution(self, current_state_distribution: np.ndarray) -> np.ndarray:
        """
        Compute the distribution derivative of the Markov chain.
        This is the change in the state distribution over a small time step.
        """
        return current_state_distribution @ self.rate_matrix


class SimpleMarkovChainQueueModel(LimitedSizeMarkovChainQueueModel):

    def __init__(self,
                 arrival_rate: float,
                 service_rate: float,
                 size: int):
        super().__init__(arrival_rate, service_rate, size)
        rate_sum = self.arrival_rate + self.service_rate
        self.increase_prob = self.arrival_rate / rate_sum
        self.decrease_prob = self.arrival_rate / rate_sum
        self.arrival_service = self.arrival_rate / self.service_rate

    @cached_property
    def stationary_distribution(self) -> np.ndarray:
        """
        Compute the continuous stationary distribution of the Markov chain.
        The continuous stationary distribution is a vector of size (size + 1)
        where each element corresponds to the probability of being in that state.
        """
        if self.arrival_service == 1:
            pi_zero = 1 / (self.size + 1)
        else:
            pi_zero = (1-self.arrival_service) / \
                (1 - (self.arrival_service**(self.size + 1)))

        pi = np.empty(self.size + 1)
        pi[0] = pi_zero
        for i in range(1, self.size + 1):
            pi[i] = self.arrival_service * pi[i - 1]

        return pi

    @cached_property
    def rate_matrix(self):
        rate_matrix = np.zeros(
            (self.size + 1, self.size + 1))  # States from 0 to size
        states = np.arange(self.size + 1)  # States from 0 to size
        rate_matrix[0, 0] = -self.arrival_rate
        rate_matrix[-1, -1] = -self.service_rate
        rate_matrix[states[1:-1], states[1:-1]] = - \
            (self.arrival_rate + self.service_rate)

        rate_matrix[states[:-1], states[1:]] = self.arrival_rate
        rate_matrix[states[1:], states[:-1]] = self.service_rate
        return rate_matrix


def total_variation(u, v):
    return .5*np.sum(np.abs(u-v))


class ArrivalRateEstimator(ABC):

    @abstractmethod
    def __call__(self,
                 time: float,
                 arrival_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        """
        Estimate the value based on the current time, arrival rate and queue size distribution.
        """
        return NotImplementedError("This method should be implemented by subclasses.")


class PassThroughEstimator(ArrivalRateEstimator):
    def __init__(self):
        pass

    def __call__(self, time: float,
                 arrival_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        return arrival_rate


class Controller(ABC):
    @abstractmethod
    def __call__(self, time: float,
                 estimated_arrival_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        """
        Compute the desired arrival rate based on the estimated service rate and the
        queue size distribution.
        """
        return NotImplementedError("This method should be implemented by subclasses.")


class ProportionalGainController(Controller):

    def __init__(self,
                 gain: float):
        self.gain = gain

    def __call__(self, time: float,
                 estimated_arrival_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        return self.gain * estimated_arrival_rate


class CloudDynamics(ABC):
    @abstractmethod
    def __call__(self, time: float,
                 requested_service_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        """
        Compute the cloud dynamics based on the desired service rate and the
        queue size distribution.
        """
        return NotImplementedError("This method should be implemented by subclasses.")


class InstantaneousCloudDynamics(CloudDynamics):
    def __init__(self):
        pass

    def __call__(self, time: float,
                 requested_service_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        return requested_service_rate


class DeployDynamics(ABC):
    @abstractmethod
    def __call__(self, time: float,
                 cloud_service_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        """
        Compute the deploy dynamics based on the cloud service rate and the
        queue size distribution.
        """
        return NotImplementedError("This method should be implemented by subclasses.")


class InstantaneousDeployDynamics(DeployDynamics):
    def __init__(self):
        pass

    def __call__(self, time: float,
                 cloud_service_rate: float,
                 queue_size_distribution: np.ndarray) -> float:
        return cloud_service_rate


class ArrivalRateDynamics(ABC):
    @abstractmethod
    def __call__(self, time: float,
                 queue_size_distribution: np.ndarray) -> float:
        """
        Compute the arrival rate based on the queue size distribution.
        """
        return NotImplementedError("This method should be implemented by subclasses.")


def get_lambda_from_string(str_fun: str,
                           variable: str = 'x'
                           ) -> Callable[[float], float]:
    sympy_locals = dict(t=sympy.Symbol(variable, real=True))
    sym_func = sympy.sympify(
        str_fun, locals=sympy_locals)
    return lambdify(variable, sym_func, modules="numpy")


class TimeFunctionArrivalRateDynamics(ArrivalRateDynamics):
    def __init__(self, arrival_rate_function: Callable[[float], float] | str):
        self.arrival_rate_function = arrival_rate_function
        if isinstance(self.arrival_rate_function, str):
            self.arrival_rate_callable = get_lambda_from_string(arrival_rate_function, "t")
        else:
            self.arrival_rate_callable = self.arrival_rate_function

    def __call__(self, time: float,
                 queue_size_distribution: np.ndarray) -> float:
        return self.arrival_rate_callable(time)


class Simulation:
    def __init__(self,
                 initial_queue_size_distribution: np.ndarray,
                 queue_size: int,
                 queue_builder: Callable[[float, float, float], LimitedSizeMarkovChainQueueModel],
                 controller: Controller,
                 cloud_dynamics: CloudDynamics,
                 deploy_dynamics: DeployDynamics,
                 arrival_rate_dynamics: ArrivalRateDynamics,
                 arrtival_rate_estimator: ArrivalRateEstimator,
                 arrival_rate_revenue: float,
                 cloud_service_rate_cost: float,
                 solve_ivp_kwargs: Dict[str, Any] = {},):
        self.initial_queue_size_distribution = initial_queue_size_distribution
        self.queue_size = queue_size
        self.queue_builder = queue_builder
        self.controller = controller
        self.cloud_dynamics = cloud_dynamics
        self.deploy_dynamics = deploy_dynamics
        self.arrival_rate_dynamics = arrival_rate_dynamics
        self.arrtival_rate_estimator = arrtival_rate_estimator
        self.arrival_rate_revenue = arrival_rate_revenue
        self.cloud_service_rate_cost = cloud_service_rate_cost
        self.solve_ivp_kwargs = solve_ivp_kwargs

        if self.solve_ivp_kwargs.get('method') is None:
            self.solve_ivp_kwargs['method'] = 'RK45'

        if self.solve_ivp_kwargs.get('max_step') is None:
            self.solve_ivp_kwargs['max_step'] = 1e-2

    def operation_cost(self, cloud_service_rate: float) -> float:
        return self.cloud_service_rate_cost * cloud_service_rate

    def operation_revenue(self, effective_arrival_rate: float) -> float:
        return self.arrival_rate_revenue * effective_arrival_rate

    def dx(self, time: float, x: np.ndarray):
        queue_size_distribution = x[:self.queue_size + 1]

        arrival_rate = self.arrival_rate_dynamics(
            time, queue_size_distribution)

        estimated_arrival_rate = self.arrtival_rate_estimator(
            time, arrival_rate, queue_size_distribution)

        requested_service_rate = self.controller(
            time, estimated_arrival_rate, queue_size_distribution)

        cloud_service_rate = self.cloud_dynamics(
            time, requested_service_rate, queue_size_distribution)

        deploy_service_rate = self.deploy_dynamics(
            time, cloud_service_rate, queue_size_distribution)

        effective_arrival_rate = LimitedSizeMarkovChainQueueModel.get_effective_arrival_rate(
            arrival_rate=arrival_rate,
            queue_size_distribution=queue_size_distribution
        )

        cost = self.operation_cost(
            cloud_service_rate=cloud_service_rate
        )
        revenue = self.operation_revenue(
            effective_arrival_rate=effective_arrival_rate
        )

        queue_model = self.queue_builder(
            arrival_rate=arrival_rate,
            service_rate=deploy_service_rate,
            size=self.queue_size
        )
        dqueue_size_distribution = queue_model.diff_distribution(
            queue_size_distribution)

        dx = np.concatenate([dqueue_size_distribution, [cost, revenue]])
        return dx

    def run(self, seconds: float) -> pd.DataFrame:

        integration_solution = integrate.solve_ivp(
            fun=self.dx,
            t_span=(0, seconds),
            y0=np.concatenate(
                [self.initial_queue_size_distribution, [0.0, 0.0]]),
            **self.solve_ivp_kwargs
        )
        columns = [
            'time',
            'arrival_rate',
            'estimated_arrival_rate',
            'effective_arrival_rate',
            'requested_service_rate',
            'cloud_service_rate',
            'deploy_service_rate',
            'cost',
            'revenue',
            'profit',
            'expected_queue_size',
            'expected_wait_time',
            'integrated_cost',
            'integrated_revenue',
            'integrated_profit',
        ]
        columns += [f'queue_size_prob_{i}' for i in range(self.queue_size + 1)]
        data = np.empty((len(integration_solution.t), len(columns)))
        results = pd.DataFrame(data,
                               columns=columns)
        for i in range(self.queue_size + 1):
            results[f'queue_size_prob_{i}'] = integration_solution.y[i, :]
        results['integrated_cost'] = integration_solution.y[-2, :]
        results['integrated_revenue'] = integration_solution.y[-1, :]
        results['time'] = integration_solution.t
        results['arrival_rate'] = np.nan
        results['estimated_arrival_rate'] = np.nan
        results['effective_arrival_rate'] = np.nan
        results['requested_service_rate'] = np.nan
        results['cloud_service_rate'] = np.nan
        results['deploy_service_rate'] = np.nan
        results['cost'] = np.nan
        results['revenue'] = np.nan
        results['expected_queue_size'] = np.nan
        results['expected_wait_time'] = np.nan
        results['profit'] = np.nan
        results['integrated_profit'] = np.nan

        for i in results.index:
            time = integration_solution.t[i]
            queue_size_distribution = integration_solution.y[:self.queue_size+1, i]
            results.loc[i, 'arrival_rate'] = self.arrival_rate_dynamics(
                time,
                queue_size_distribution
            )
            results.loc[i, 'estimated_arrival_rate'] = self.arrtival_rate_estimator(
                time,
                results.loc[i, 'arrival_rate'],
                queue_size_distribution
            )
            results.loc[i, 'requested_service_rate'] = self.controller(
                time,
                results.loc[i, 'estimated_arrival_rate'],
                queue_size_distribution
            )
            results.loc[i, 'cloud_service_rate'] = self.cloud_dynamics(
                time,
                results.loc[i, 'requested_service_rate'],
                queue_size_distribution
            )
            results.loc[i, 'deploy_service_rate'] = self.deploy_dynamics(
                time,
                results.loc[i, 'cloud_service_rate'],
                queue_size_distribution
            )
            results.loc[i, 'expected_queue_size'] = LimitedSizeMarkovChainQueueModel.get_expected_size(
                queue_size_distribution=queue_size_distribution
            )
            results.loc[i, 'effective_arrival_rate'] = LimitedSizeMarkovChainQueueModel.get_effective_arrival_rate(
                arrival_rate=results.loc[i, 'arrival_rate'],
                queue_size_distribution=queue_size_distribution
            )
            results.loc[i, 'expected_wait_time'] = LimitedSizeMarkovChainQueueModel.get_expected_wait_time(
                expected_size=results.loc[i, 'expected_queue_size'],
                effective_arrival_rate=results.loc[i, 'effective_arrival_rate']
            )
            results.loc[i, 'cost'] = self.operation_cost(
                cloud_service_rate=results.loc[i, 'cloud_service_rate']
            )
            results.loc[i, 'revenue'] = self.operation_revenue(
                effective_arrival_rate=results.loc[i, 'effective_arrival_rate']
            )
            results.loc[i, 'profit'] = results.loc[i,
                                                   'revenue'] - results.loc[i, 'cost']
            results.loc[i, 'integrated_profit'] = results.loc[i,
                                                              'integrated_revenue'] - results.loc[i, 'integrated_cost']

        return results


class OptimizedControl:

    def __init__(self,
                 initial_queue_size_distribution: np.ndarray,
                 controller_builder: Callable[[float], Controller],
                 queue_builder: Callable[[float, float, float], LimitedSizeMarkovChainQueueModel],
                 cloud_dynamics: CloudDynamics,
                 deploy_dynamics: DeployDynamics,
                 arrival_rate_dynamics: ArrivalRateDynamics,
                 arrtival_rate_estimator: ArrivalRateEstimator,
                 arrival_rate_revenue: float,
                 cloud_service_rate_cost: float,
                 seconds: float,
                 output_dir: str | Path | None,
                 solve_ivp_kwargs: Dict[str, Any] = {},
                 minimize_kwargs: Dict[str, Any] = {}):
        self.initial_queue_size_distribution = initial_queue_size_distribution
        self.controller_builder = controller_builder
        self.queue_builder = queue_builder
        self.cloud_dynamics = cloud_dynamics
        self.deploy_dynamics = deploy_dynamics
        self.arrival_rate_dynamics = arrival_rate_dynamics
        self.arrtival_rate_estimator = arrtival_rate_estimator
        self.arrival_rate_revenue = arrival_rate_revenue
        self.cloud_service_rate_cost = cloud_service_rate_cost
        self.seconds = seconds
        self.solve_ivp_kwargs = solve_ivp_kwargs
        self.minimize_kwargs = minimize_kwargs

        if not output_dir:
            self.output_dir = None
        if isinstance(output_dir, str):
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = output_dir

        self._reset_records()

    def _reset_records(self):
        self._call_counter = 0
        self._result = None
        self._costs: List[float] = []
        self._controller_params: List[float] = []
        self._initial_controller_params: np.ndarray = []
        self._solution: np.ndarray = []

    def cost_function(self, controller_params: np.ndarray) -> float:
        logging.info(
            f"{self._call_counter} - Running cost function with controller params: {controller_params}")
        simulation = Simulation(
            initial_queue_size_distribution=self.initial_queue_size_distribution,
            queue_size=len(self.initial_queue_size_distribution) - 1,
            queue_builder=self.queue_builder,
            controller=self.controller_builder(*controller_params),
            cloud_dynamics=self.cloud_dynamics,
            deploy_dynamics=self.deploy_dynamics,
            arrival_rate_dynamics=self.arrival_rate_dynamics,
            arrtival_rate_estimator=self.arrtival_rate_estimator,
            arrival_rate_revenue=self.arrival_rate_revenue,
            cloud_service_rate_cost=self.cloud_service_rate_cost,
            solve_ivp_kwargs=self.solve_ivp_kwargs
        )
        simulation_result = simulation.run(seconds=self.seconds)
        if self.output_dir:
            simulation_result.to_parquet(self.output_dir / f'simulation_{self._call_counter:03}.parquet',
                                         engine='pyarrow')
        self._call_counter += 1
        cost = -simulation_result.iloc[-1]['integrated_profit']
        self._controller_params.append(controller_params.tolist())
        self._costs.append(float(cost))
        return cost

    def optimize(self, initial_controller_params: np.ndarray) -> Dict[str, Any]:
        self._reset_records()
        self._initial_controller_params = initial_controller_params.copy()
        numpy_result = optimize.minimize(
            self.cost_function,
            x0=initial_controller_params,
            method='Nelder-Mead',
            **self.minimize_kwargs
        )
        self._set_numpy_result(numpy_result)
        return self._result

    def _set_numpy_result(self, result):
        self._solution = result.x.tolist()
        self._result = dict(
            x=self._solution,
            fun=float(result.fun),
            status=result.status,
            success=result.success,
            message=result.message,
            nfev=result.nfev,
            nit=result.nit
        )

    def as_dict(self):
        return dict(
            controller_params=self._controller_params,
            costs=self._costs,
            initial_queue_size_distribution=self.initial_queue_size_distribution.tolist(),
            initial_controller_params=self._initial_controller_params.tolist(),
            cloud_service_rate_cost=float(self.cloud_service_rate_cost),
            arrival_rate_revenue=float(self.arrival_rate_revenue),
            seconds=float(self.seconds),
            call_counter=self._call_counter,
            result=self._result,
            solve_ivp_kwargs=self.solve_ivp_kwargs,
            minimize_kwargs=self.minimize_kwargs
        )

    def dump_results(self):
        if not self.output_dir:
            raise ValueError(
                "Output directory is not set. Cannot dump results.")
        filename = os.path.join(self.output_dir, 'optimization_results.json')
        with open(filename, 'w') as f:
            json.dump(self.as_dict(), f, indent=4)


class OptimizedControlResults:
    controller_params: List[List[float]]
    costs: List[float]
    initial_queue_size_distribution: List[float]
    initial_controller_params: List[float]
    cloud_service_rate_cost: float
    arrival_rate_revenue: float
    seconds: float
    call_counter: int
    result: Dict[str, Any]
    solve_ivp_kwargs: Dict[str, Any]
    minimize_kwargs: Dict[str, Any]

    def __init__(self,
                 controller_params: ArrayLike,
                 costs: ArrayLike,
                 initial_queue_size_distribution: ArrayLike,
                 initial_controller_params: ArrayLike,
                 cloud_service_rate_cost: float,
                 arrival_rate_revenue: float,
                 seconds: float,
                 call_counter: int,
                 result: Dict[str, Any],
                 solve_ivp_kwargs: Dict[str, Any] = {},
                 minimize_kwargs: Dict[str, Any] = {}):
        if not isinstance(controller_params, np.ndarray):
            self.controller_params = np.array(controller_params)
        if not isinstance(costs, np.ndarray):
            self.costs = np.array(costs)
        if not isinstance(initial_queue_size_distribution, np.ndarray):
            self.initial_queue_size_distribution = np.array(
                initial_queue_size_distribution)
        if not isinstance(initial_controller_params, np.ndarray):
            self.initial_controller_params = np.array(initial_controller_params)
        self.cloud_service_rate_cost = cloud_service_rate_cost
        self.arrival_rate_revenue = arrival_rate_revenue
        self.seconds = seconds
        self.call_counter = call_counter
        self.result = result
        self.solve_ivp_kwargs = solve_ivp_kwargs
        self.minimize_kwargs = minimize_kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizedControlResults':
        return cls(
            controller_params=data['controller_params'],
            costs=data['costs'],
            initial_queue_size_distribution=data['initial_queue_size_distribution'],
            initial_controller_params=data['initial_controller_params'],
            cloud_service_rate_cost=data['cloud_service_rate_cost'],
            arrival_rate_revenue=data['arrival_rate_revenue'],
            seconds=data['seconds'],
            call_counter=data['call_counter'],
            result=data['result'],
            solve_ivp_kwargs=data['solve_ivp_kwargs'],
            minimize_kwargs=data['minimize_kwargs']
        )

    @classmethod
    def from_json(cls, filename: str | Path) -> 'OptimizedControlResults':
        if isinstance(filename, str):
            filename = Path(filename)
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @cached_property
    def best_iteration(self) -> int:
        return np.argmin(self.costs)

    @property
    def best_controller_params(self) -> float:
        return self.controller_params[self.best_iteration]

    @property
    def best_cost(self) -> float:
        return self.costs[self.best_iteration]

    @property
    def queue_size(self) -> int:
        return len(self.initial_queue_size_distribution) - 1


def normal_distribution_str(variable: str, mean: float, std: float) -> str:
    """
    Generate a string representation of a normal distribution.
    """
    return f"((exp(-((({variable}-{mean})/{std})^2)))/({std}*sqrt(2*pi)))"
