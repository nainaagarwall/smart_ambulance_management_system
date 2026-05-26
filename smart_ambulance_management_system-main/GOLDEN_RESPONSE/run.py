#!/usr/bin/env python
"""
CLI entrypoint to execute the Intelligent Emergency Response Optimization System.
"""

import sys
import argparse
import logging
from src.pipeline import EmergencyResponsePipeline

def main():
    parser = argparse.ArgumentParser(
        description="Python-Based Intelligent Emergency Response Optimization System (EROS)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generating synthetic dataset and only run on existing files"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set logging level to DEBUG for verbose outputs"
    )
    
    args = parser.parse_args()
    
    # Setup base logging for startup
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("EROS_CLI")
    
    logger.info("Initializing Emergency Response Optimization System...")
    
    try:
        pipeline = EmergencyResponsePipeline(config_path=args.config)
        
        # Override log level if debug is requested on CLI
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Debug logging enabled.")
            
        eval_report, report_path = pipeline.run_pipeline(generate_data=not args.no_generate)
        
        logger.info("Pipeline execution completed successfully!")
        logger.info("Model evaluation report saved at: %s", report_path)
        logger.info("JSON dispatch alerts and CSV admin summaries are ready under 'outputs/reports/'.")
        logger.info("Analytics charts are saved under 'outputs/plots/'.")
        
    except Exception as e:
        logger.critical("Fatal exception during pipeline execution: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
