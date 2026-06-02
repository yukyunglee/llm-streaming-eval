"""
Configuration management for the unified streaming pipeline.
"""

import os
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    # Data settings
    'max_input_tokens': 4000,
    'texts_per_event': 1,
    'random_seed': 42,
    
    # Window settings
    'default_window_size': 7,
    'default_stride': 1,
    'window_type': 'sliding',
    
    # LLM settings
    'default_engine': 'mock',
    'max_retries': 3,
    'timeout_seconds': 30,
    
    # Output settings
    'output_dir': 'results',
    'save_results': True,
    'log_level': 'INFO',
    
    # Task-specific settings
    'clustering': {
        'similarity_threshold': 0.8,
        'min_cluster_size': 2,
        'embedding_model': 'all-MiniLM-L6-v2'
    },
    'summarization': {
        'max_summary_length': 200,
        'min_summary_length': 50
    },
    'temporal_qa': {
        'max_qa_pairs': 5,
        'sample_rate': 0.5
    }
}

# Environment variable mappings
ENV_VARS = {
    'TOGETHER_API_KEY': 'together_api_key',
    'OPENAI_API_KEY': 'openai_api_key',
    'VLLM_BASE_URL': 'vllm_base_url',
    'OUTPUT_DIR': 'output_dir',
    'LOG_LEVEL': 'log_level'
}


class Config:
    """
    Configuration manager for the unified pipeline.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        
        # Load from environment variables
        self._load_from_env()
        
        # Override with provided config
        if config_dict:
            self._deep_update(self.config, config_dict)
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        for env_var, config_key in ENV_VARS.items():
            value = os.getenv(env_var)
            if value:
                self.config[config_key] = value
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()
    
    def validate_api_keys(self, engine: str) -> bool:
        """
        Validate that required API keys are available.
        
        Args:
            engine: LLM engine name
            
        Returns:
            True if API keys are valid
        """
        if engine == 'together':
            return bool(self.get('together_api_key'))
        elif engine == 'openai':
            return bool(self.get('openai_api_key'))
        elif engine == 'vllm':
            return bool(self.get('vllm_base_url'))
        elif engine == 'mock':
            return True
        else:
            return False
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get task-specific configuration."""
        return self.get(task_name, {})
    
    def print_config(self):
        """Print current configuration."""
        print("Current Configuration:")
        print("=" * 50)
        
        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                else:
                    # Hide API keys for security
                    if 'api_key' in key.lower():
                        display_value = "*" * 10 if value else "Not set"
                    else:
                        display_value = value
                    print("  " * indent + f"{key}: {display_value}")
        
        print_dict(self.config)
        print("=" * 50)


# Global configuration instance
_config = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config
