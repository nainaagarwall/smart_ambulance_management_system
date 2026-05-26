"""
Configuration loader and logging setup for the Intelligent Emergency Response Optimization System.
"""

import os
import yaml
import logging
from typing import Dict, Any

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configures logging for the application.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set default values if keys are missing
    config.setdefault('system', {})
    config['system'].setdefault('random_seed', 42)
    config['system'].setdefault('log_level', 'INFO')
    
    # Initialize logging
    setup_logging(config['system']['log_level'])
    logger = logging.getLogger(__name__)
    logger.debug("Configuration loaded successfully from %s", config_path)
    
    return config
