from functools import lru_cache
import numpy as np
from utils import (
    OptimizedControl,
    SimpleMarkovChainQueueModel,
    PassThroughEstimator,
    InstantaneousCloudDynamics,
    InstantaneousDeployDynamics,
    ProportionalGainController,
    TimeFunctionArrivalRateDynamics,
    normal_distribution_str

)
from pathlib import Path
import logging
import logging.config
from joblib import Parallel, delayed


def get_logging_config(log_level: str,
                       output_dilename: str) -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(module)s"
                " | %(lineno)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": log_level,
                "formatter": "default",
                "filename": output_dilename,
                "mode": "a"
            }
        },
        "loggers": {
            "root": {
                "level": log_level,
                "handlers": ["file"]
            }
        }
    }


def run_optimization(job_id: int,
                     queue_size: int,
                     arrival_rate_function: str,
                     arrival_rate_revenue: float,
                     cloud_service_rate_cost: float,
                     xatol: float = 1e-5,
                     seconds: float = 600,
                     base_output_dir: str = '.') -> dict:

    @lru_cache(maxsize=10)
    def get_queue_model(arrival_rate: float, service_rate: float, size: int) -> SimpleMarkovChainQueueModel:
        return SimpleMarkovChainQueueModel(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            size=size
        )

    output_dir = Path(base_output_dir) / f'job_{job_id}'
    if output_dir.exists():
        print(f'Output directory {output_dir} already exists. Skipping job {job_id}.')
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_filename = output_dir / 'job.log'
    logging.config.dictConfig(get_logging_config('INFO', str(log_filename)))
    initial_queue_size_distribution = np.zeros(queue_size + 1, dtype=float)
    initial_queue_size_distribution[0] = 1.0
    optimizer = OptimizedControl(
        initial_queue_size_distribution=initial_queue_size_distribution,
        controller_builder=ProportionalGainController,
        queue_builder=get_queue_model,
        cloud_dynamics=InstantaneousCloudDynamics(),
        deploy_dynamics=InstantaneousDeployDynamics(),
        arrival_rate_dynamics=TimeFunctionArrivalRateDynamics(
            arrival_rate_function),
        arrtival_rate_estimator=PassThroughEstimator(),
        arrival_rate_revenue=arrival_rate_revenue,
        cloud_service_rate_cost=cloud_service_rate_cost,
        seconds=seconds,
        output_dir=output_dir,
        solve_ivp_kwargs={
            'method': 'RK45',
            'max_step': 1e-1,
        },
        minimize_kwargs={
            'bounds': [(xatol, np.inf)],
            'options': {
                'maxiter': 100,
                'xatol': xatol
            }
        }
    )
    initial_controller_params = np.array([1.])
    logging.info(f'{job_id} - Starting optimization')
    logging.info(f'arrival_rate_function: {arrival_rate_function}')
    logging.info(f'optimizer.initial_queue_size_distribution: {optimizer.initial_queue_size_distribution}')
    logging.info(f'optimizer.controller_builder: {optimizer.controller_builder}')
    logging.info(f'optimizer.arrival_rate_revenue: {optimizer.arrival_rate_revenue}')
    logging.info(f'optimizer.cloud_service_rate_cost: {optimizer.cloud_service_rate_cost}')
    logging.info(f'optimizer.seconds: {optimizer.seconds}')
    logging.info(f'optimizer.solve_ivp_kwargs: {optimizer.solve_ivp_kwargs}')
    logging.info(f'optimizer.minimize_kwargs: {optimizer.minimize_kwargs}')
    logging.info(f'optimizer.output_dir: {optimizer.output_dir}')
    logging.info(f'Initial controller parameters: {initial_controller_params}')
    logging.info('Optimization started')
    optimizer.optimize(initial_controller_params)
    logging.info('Optimization completed')
    logging.info('Optimization results:')
    results = optimizer.as_dict()
    results['arrival_rate_function'] = arrival_rate_function
    for key, value in results.items():
        logging.info(f'{key}: {value}')
    optimizer.dump_results()
    logging.info(f'Results saved to {output_dir}')


if __name__ == '__main__':
    print("Starting simulation script")
    base_output_dir = Path('simulation_results')
    queue_size = 100
    mean = 300
    std = 100
    params = [
        {
            "job_id": 'cheap_cloud_constant_demand',
            "queue_size": queue_size,
            "arrival_rate_function": "10",
            "arrival_rate_revenue": 1.0,
            "cloud_service_rate_cost": 0.1,
            "base_output_dir": str(base_output_dir),
        },
        {
            "job_id": 'expensive_cloud_constant_demand',
            "queue_size": queue_size,
            "arrival_rate_function": "10",
            "arrival_rate_revenue": 1.0,
            "cloud_service_rate_cost": 0.5,
            "base_output_dir": str(base_output_dir),
        },
        {
            "job_id": 'cheap_cloud_normal_dist_demand',
            "queue_size": queue_size,
            "arrival_rate_function": f"10*(1+250*{normal_distribution_str("t", mean, std)})",
            "arrival_rate_revenue": 1.0,
            "cloud_service_rate_cost": 0.1,
            "base_output_dir": str(base_output_dir),
        },
        {
            "job_id": 'expensive_cloud_normal_dist_demand',
            "queue_size": queue_size,
            "arrival_rate_function": f"10*(1+250*{normal_distribution_str("t", mean, std)})",
            "arrival_rate_revenue": 1.0,
            "cloud_service_rate_cost": 0.5,
            "base_output_dir": str(base_output_dir),
        },
    ]

    pool = Parallel(n_jobs=-1)
    pool(delayed(run_optimization)(**param) for param in params)
    print("Simulation script completed")
