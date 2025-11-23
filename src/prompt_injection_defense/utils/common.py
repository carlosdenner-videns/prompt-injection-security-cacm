"""Common utility functions for data loading, configuration, and I/O."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict


def load_json(filepath: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded JSON data as dict
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output path
        indent: JSON indentation (default: 2)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_config(filepath: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        filepath: Path to YAML config file
        
    Returns:
        Configuration as dict
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent.parent
