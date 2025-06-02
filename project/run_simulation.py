from argparse import ArgumentParser
from utils import Simulation
from pathlib import Path
from datetime import datetime
import logging
import logging.config


def parse_args():

    parser = ArgumentParser()

    parser.add_argument('--seconds',
                        required=True,
                        type=float,
                        help='Number of seconds to run the simulation')
    parser.add_argument('--initial-size',
                        required=True,
                        type=int,
                        dest='initial_size',
                        help='Queue initial size')
    parser.add_argument('--queue-size',
                        required=True,
                        type=int,
                        dest='queue_size',
                        help='Queue max size')
    parser.add_argument('--time-step',
                        required=True,
                        type=float,
                        dest='time_step',
                        help='Simulation time step in seconds')
    parser.add_argument('--window',
                        required=True,
                        type=float,
                        dest='window',
                        help='Simulation time step in seconds')
    parser.add_argument('--infra-cost',
                        required=True,
                        type=float,
                        dest='infra_cost',
                        help='The infra cost multiplier')
    parser.add_argument('--wait-time-cost',
                        required=True,
                        type=float,
                        dest='wait_time_cost',
                        help='The infra cost multiplier')
    parser.add_argument('--customer-revenue',
                        required=True,
                        type=float,
                        dest='costumer_revenue',
                        help='Money earned per customer')

    return parser.parse_args()


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
    # args = parse_args()
    # sim = Simulation(
    #     initial_queue_size=args.initial_size,
    #     queue_size=args.queue_size,
    #     time_step=args.time_step,
    #     producer_lambda_function=None,
    #     controller_window=args.window,
    #     infra_cost=args.infra_cost,
    #     wait_time_cost=args.wait_time_cost,
    #     customer_revenue=args.customer_revenue
    # )
    logging.config.dictConfig(get_logging_config())
    output_dir = Path(f'/home/lucasbanunes/workspaces/monte-carlo-cps767/CPS767-monte-carlo-algorithms/project/simulation_results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)
    sim = Simulation(
        initial_queue_size=0,
        queue_size=100,
        time_step=1e-3,
        producer_lambda_function=lambda x: 10,
        controller_window=1,
        infra_cost=10,
        wait_time_cost=2,
        customer_revenue=20
    )

    output_file = output_dir / 'data.parquet'
    sim.run(600)
    sim.dump(str(output_file))
