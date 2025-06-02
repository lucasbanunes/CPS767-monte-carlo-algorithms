import numpy as np
from utils import (
    TimeContinuousMarkovChainQueueModel,
    total_variation,
)
from functools import cache
from typing import List
import pandas as pd
from pathlib import Path
from datetime import datetime


@cache
def get_queue_model(producer_lambda: float,
                    consumer_lambda: float,
                    queue_size: int,
                    discrete_time_step: float = 0.01) -> TimeContinuousMarkovChainQueueModel:
    return TimeContinuousMarkovChainQueueModel(
        producer_lambda=producer_lambda,
        consumer_lambda=consumer_lambda,
        size=queue_size,
        time_step=discrete_time_step
    )


def get_fail_prob(distribution: np.ndarray) -> float:
    """
    Calculate the failure probability based on the distribution.
    The failure probability is the last element of the distribution.
    """
    return distribution[-1]


def get_expected_size(distribution: np.ndarray) -> float:
    """
    Calculate the expected state based on the distribution.
    """
    return np.dot(np.arange(len(distribution)), distribution)


def get_expected_wait_time(effective_producer_lambda: float,
                           expected_state: float) -> float:
    return effective_producer_lambda * expected_state


start_producer_lambda = 1000
producer_lambda_multiplier = 2


def get_producer_lambda(time: float):
    return producer_lambda_multiplier * start_producer_lambda


def get_effective_producer_lambda(producer_lambda: float,
                                  fail_prob: float) -> float:
    """
    Calculate the effective producer lambda based on the producer lambda
    and the failure probability.
    """
    return producer_lambda * (1 - fail_prob)


start_machines = np.array([0, 1, 0, 0, 1])


def get_active_machines(time: float):
    if time == 0:
        return start_machines
    else:
        return producer_lambda_multiplier*start_machines


increase_response_time = 120
end_machines = producer_lambda_multiplier*start_machines


def get_processing_machines(time: float):
    if time > increase_response_time:
        return end_machines
    return start_machines


multiplier = np.arange(5)
base_price = 1
operation_cost = 1*2*multiplier


def get_infra_cost(active_machines: np.ndarray):
    return np.dot(operation_cost, active_machines)


base_lambda = 100
machines_lambda = base_lambda*2.1*multiplier


def get_consumer_lambda(processing_machines: np.ndarray):
    return np.dot(machines_lambda, processing_machines)


wait_time_cost = 0.01


def get_cost(infra_cost: float,
             expected_wait_time: float):
    return infra_cost + wait_time_cost*expected_wait_time


customer_revenue = 0.01


def get_revenue(effective_producer_lambda: float):
    return customer_revenue * effective_producer_lambda


start_consumer_lambda = get_consumer_lambda(0)


cooldown = 100
current_step = 0
current_time = 0
current_variation = np.inf
current_distribution = None
epsilon = 1e-2
max_steps = int(1e6)

queue_size = 100
discrete_time_step = 0.01

current_step = 0
current_time = 0

times: List[float] = []
distributions: List[np.ndarray] = []
variations: List[float] = []
fail_probs: List[float] = []
expected_sizes: List[float] = []
producer_lambdas: List[float] = []
effective_producer_lambdas: List[float] = []
expected_wait_times: List[float] = []
consumer_lambdas: List[float] = []
infra_costs: List[float] = []
costs: List[float] = []
revenues: List[float] = []
profits: List[float] = []

while current_step < max_steps and (cooldown > current_step or current_variation > epsilon):

    print(f'Running step {current_step}, time {current_time}, variation {current_variation}')
    times.append(current_time)
    current_producer_lambda = get_producer_lambda(current_time)
    processing_machines = get_processing_machines(current_time)
    current_consumer_lambda = get_consumer_lambda(processing_machines)
    consumer_lambdas.append(current_consumer_lambda)

    current_model = get_queue_model(
        current_producer_lambda,
        current_consumer_lambda,
        queue_size
    )
    if current_time:
        current_distributions = current_model.discrete.update_distribution(
            current_distribution
        )
    else:
        current_distribution = current_model.discrete.lstsq_distribution
    distributions.append(current_distribution)
    current_variation = total_variation(
        current_distribution,
        current_model.discrete.lstsq_distribution
    )
    variations.append(current_variation)
    fail_prob = current_distribution[-1]
    fail_probs.append(fail_prob)
    expected_size = get_expected_size(current_distribution)
    expected_sizes.append(expected_size)
    producer_lambdas.append(current_producer_lambda)
    effective_producer_lambda = get_effective_producer_lambda(
        current_producer_lambda, fail_prob
    )
    effective_producer_lambdas.append(effective_producer_lambda)
    expected_wait_time = get_expected_wait_time(
        effective_producer_lambda,
        expected_size
    )
    expected_wait_times.append(expected_wait_time)
    active_machines = get_active_machines(current_time)
    infra_cost = get_infra_cost(active_machines)
    infra_costs.append(infra_cost)
    cost = get_cost(
        infra_cost,
        expected_wait_time
    )
    costs.append(cost)
    revenue = get_revenue(
        effective_producer_lambda)
    revenues.append(revenue)
    profit = revenue - cost
    profits.append(profit)

    current_step += 1
    current_time += discrete_time_step

result_df = dict(
    time=times,
    distribution=distributions,
    variation=variations,
    fail_prob=fail_probs,
    expected_size=expected_sizes,
    producer_lambda=producer_lambdas,
    effective_producer_lambda=effective_producer_lambdas,
    expected_wait_time=expected_wait_times,
    consumer_lambda=consumer_lambdas,
    infra_cost=infra_costs,
    cost=costs,
    revenue=revenues,
    profit=profits
)

output_dir = Path(f'/home/lucasbanunes/workspaces/monte-carlo-cps767/CPS767-monte-carlo-algorithms/project/simulation_results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
output_dir.mkdir(parents=True, exist_ok=True)
result_df = pd.DataFrame.from_dict(result_df)
output_file = output_dir / 'data.parquet'
result_df.to_parquet(output_file)
