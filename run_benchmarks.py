"""Main script to run comprehensive benchmarks."""

import os
import logging
from benchmarks.comparison_benchmark import ComprehensiveBenchmark

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create benchmarks directory if it doesn't exist
    os.makedirs('benchmarks/results', exist_ok=True)
    
    # Run benchmarks
    logger.info("Starting comprehensive benchmarks...")
    benchmark = ComprehensiveBenchmark()
    
    results = benchmark.run_benchmarks(num_tasks=20)
    benchmark.plot_results(results)
    
    # Log results
    logger.info("\nBenchmark Results:")
    for approach, metrics in results.items():
        logger.info(f"\n{approach.upper()} RESULTS:")
        for component, data in metrics.items():
            logger.info(f"\n{component.upper()}:")
            for algo, values in data.items():
                logger.info(f"{algo}:")
                for metric, value in values.items():
                    logger.info(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    main()
