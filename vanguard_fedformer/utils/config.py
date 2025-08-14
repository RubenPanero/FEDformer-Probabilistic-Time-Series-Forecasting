"""
Configuration management for Vanguard-FEDformer.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Union
import os

class ConfigManager:
    """
    Manages configuration loading and access for Vanguard-FEDformer.
    
    Supports YAML configuration files with nested attribute access.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to configuration."""
        if name in self.config:
            value = self.config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.config[key]
    
    def save(self, output_path: Union[str, Path] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Output path (defaults to original path)
        """
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._update_nested(self.config, updates)
    
    def _update_nested(self, config: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update nested configuration."""
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_nested(config[key], value)
            else:
                config[key] = value
    
    def validate(self) -> bool:
        """
        Validate configuration structure.
        
        Returns:
            True if configuration is valid
        """
        required_keys = ['model', 'data', 'training']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")
        
        return True
    
    def print_summary(self) -> None:
        """Print a summary of the configuration."""
        print(f"Configuration loaded from: {self.config_path}")
        print(f"Configuration sections: {list(self.config.keys())}")
        
        for section, config in self.config.items():
            if isinstance(config, dict):
                print(f"\n{section.upper()}:")
                for key, value in config.items():
                    print(f"  {key}: {value}")


class ConfigSection:
    """
    Represents a configuration section with attribute access.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to configuration section."""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration section has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration section."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)