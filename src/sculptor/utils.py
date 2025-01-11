from typing import Any, Dict
import os

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load and parse a configuration file with environment variable substitution.
    Supports JSON and YAML formats.
    
    Args:
        filepath: Path to config file (.json, .yaml, or .yml)
        
    Returns:
        Dict containing the parsed configuration
    """
    # Read raw config with env vars
    with open(filepath, 'r') as f:
        config_str = f.read()
    
    # Replace environment variables
    config_str = os.path.expandvars(config_str)
    
    # Parse based on file extension
    if filepath.endswith(".json"):
        import json
        return json.loads(config_str)
    elif filepath.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(config_str)
    else:
        raise ValueError(
            "Invalid file type. The config file must be a JSON or YAML file "
            "(with .json, .yaml, or .yml extension)."
        )