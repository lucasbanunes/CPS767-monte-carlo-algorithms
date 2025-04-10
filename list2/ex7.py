
from itertools import product
import string
import asyncio
import httpx
import numpy as np
import pandas as pd
from typing import List, Iterable
import logging
import logging.config
from joblib import Parallel, delayed
from pathlib import Path

DEFAULT_LOGGING_CONFIG = {
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
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
    },
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["stdout"]
        }
    }
}
logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)

async def adomain_exists(domain: str) -> int:
    """
    Asynchronous check if a domain exists in the DNS.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f'https://{domain}', timeout=5)
            return 1 if response.status_code == 200 else 0
    except (httpx.RequestError, httpx.TimeoutException):
        return 0


def domain_generator(k: int) -> Iterable[str]:
    """
    Generate all possible domain names with length up to k.
    """
    for i in range(1, k + 1):
        for domain_chars in product(string.ascii_lowercase, repeat=i):
            yield f"{''.join(domain_chars)}.ufrj.br"

async def aquery_domains(domains: Iterable[str]) -> List[int]:
    """
    Asynchronous function to query domains.
    """
    tasks = [adomain_exists(domain) for domain in domains]
    return await asyncio.gather(*tasks)

def query_domains(domains: Iterable[str], job_id: int, output_dir: Path):
    """
    Query domains in parallel using asyncio.
    """
    logging.info(f'Job {job_id} started.')
    df = pd.DataFrame(columns=['domain', 'exists'])
    df['domain'] = list(domains)
    df['exists'] = asyncio.run(aquery_domains(domains))
    output_file = output_dir / f'job_{job_id}.csv'
    df.to_csv(output_file, index=False)
    logging.info(f'Job {job_id} completed. Results saved to {output_file}.')


k=4
n_jobs = 10
output_dir = Path('ex7_output')
output_dir.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(columns=['domain', 'exists'])
results['domain'] = np.random.permutation(list(domain_generator(k)))
logging.info('Starting domain existence check...')
job_pool = Parallel(n_jobs=n_jobs)
domains_per_job = 1000
job_results = job_pool(
    delayed(query_domains)(results.loc[i:i + domains_per_job, 'domain'], job_id, output_dir)
    for job_id, i in enumerate(range(0, len(results) + domains_per_job, domains_per_job))
)
# results['exists'] = np.concatenate(job_results)
logging.info('Domain existence check completed.')
# results.to_csv('ex7.csv', index=False)