import numpy as np
from utils import Optimizer
from pathlib import Path
from datetime import datetime
import logging
import logging.config


def get_logging_config(log_level: str = 'DEBUG'):
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
        },
        "loggers": {
            "root": {
                "level": log_level,
                "handlers": ["stdout"]
            }
        }
    }


if __name__ == '__main__':
    logging.config.dictConfig(get_logging_config('INFO'))
    output_dir = Path(f'/home/lucasbanunes/workspaces/monte-carlo-cps767/CPS767-monte-carlo-algorithms/project/simulation_results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_size = 100
    initial_queue_size_distribution = np.zeros(queue_size + 1)
    initial_queue_size_distribution[0] = 1.0  # Start with an empty queue
    optimizer = Optimizer(
        initial_queue_size_distribution=initial_queue_size_distribution,
        initial_controller_gain=1.,
        producer_lambda_function="max(10, 10*(1+sign(t-40)*exp(-(((t-300)/100)^2))))",
        cache_models=True,
        queue_size=queue_size,
        infra_cost=0.1,
        wait_time_cost=0.1,
        customer_revenue=1.0,
        seconds=600,
        time_step=1e-2,
        output_dir=output_dir
    )
    logging.info('Starting optimization')
    logging.info(f'Initial queue size distribution: {optimizer.initial_queue_size_distribution}')
    logging.info(f'Initial controller gain: {optimizer.initial_controller_gain}')
    logging.info(f'Producer lambda function: {optimizer.producer_lambda_function}')
    logging.info(f'Queue size: {optimizer.queue_size}')
    logging.info(f'Infra cost: {optimizer.infra_cost}')
    logging.info(f'Wait time cost: {optimizer.wait_time_cost}')
    logging.info(f'Customer revenue: {optimizer.customer_revenue}')
    logging.info(f'Simulation duration: {optimizer.seconds} seconds')
    logging.info(f'Time step: {optimizer.time_step} seconds')
    logging.info(f'Output directory: {output_dir}')
    logging.info('Running optimization...')
    optimizer.run()
    logging.info('Optimization completed')
    logging.info('Optimization results:')
    results = optimizer.as_dict()
    for key, value in results.items():
        logging.info(f'{key}: {value}')
    optimizer.dump_results()
    logging.info(f'Results saved to {output_dir}')
