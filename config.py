import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_CONFIG_FILE = 'config.yaml'
DEFAULT_LOG_LEVEL = 'INFO'

# Define configuration class
class Config:
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.config = self.load_config()

    def load_config(self) -> Dict:
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            logger.error(f'Config file {self.config_file} not found.')
            return {}
        except yaml.YAMLError as e:
            logger.error(f'Error parsing config file {self.config_file}: {e}')
            return {}

    def save_config(self, config: Dict):
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def get_config(self, key: str, default: Optional[str] = None) -> str:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str):
        self.config[key] = value
        self.save_config(self.config)

# Define configuration manager class
class ConfigurationManager:
    def __init__(self):
        self.config = Config()

    def get_config(self, key: str, default: Optional[str] = None) -> str:
        return self.config.get_config(key, default)

    def set_config(self, key: str, value: str):
        self.config.set_config(key, value)

# Define log level enum
class LogLevel(Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'

# Define logger class
class Logger:
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level

    def debug(self, message: str):
        logger.debug(message)

    def info(self, message: str):
        logger.info(message)

    def warning(self, message: str):
        logger.warning(message)

    def error(self, message: str):
        logger.error(message)

# Define configuration class with logging
class ConfigWithLogging(Config):
    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)
        self.logger = Logger(self.get_config('log_level', DEFAULT_LOG_LEVEL))

    def get_config(self, key: str, default: Optional[str] = None) -> str:
        return super().get_config(key, default)

    def set_config(self, key: str, value: str):
        super().set_config(key, value)

# Define configuration manager class with logging
class ConfigurationManagerWithLogging(ConfigurationManager):
    def __init__(self):
        super().__init__()
        self.logger = Logger(self.get_config('log_level', DEFAULT_LOG_LEVEL))

    def get_config(self, key: str, default: Optional[str] = None) -> str:
        return super().get_config(key, default)

    def set_config(self, key: str, value: str):
        super().set_config(key, value)

# Define constants for the research paper
class ResearchPaperConstants:
    def __init__(self):
        self.velocity_threshold = 0.5
        self.flow_theory_threshold = 0.8

# Define configuration class with research paper constants
class ConfigWithResearchPaperConstants(ConfigWithLogging):
    def __init__(self, config_file: Optional[str] = None):
        super().__init__(config_file)
        self.research_paper_constants = ResearchPaperConstants()

# Define configuration manager class with research paper constants
class ConfigurationManagerWithResearchPaperConstants(ConfigurationManagerWithLogging):
    def __init__(self):
        super().__init__()
        self.research_paper_constants = ResearchPaperConstants()

# Define main configuration class
class MainConfig:
    def __init__(self):
        self.config_manager = ConfigurationManagerWithResearchPaperConstants()

    def get_config(self, key: str, default: Optional[str] = None) -> str:
        return self.config_manager.get_config(key, default)

    def set_config(self, key: str, value: str):
        self.config_manager.set_config(key, value)

# Define main configuration manager class
class MainConfigurationManager:
    def __init__(self):
        self.main_config = MainConfig()

    def get_config(self, key: str, default: Optional[str] = None) -> str:
        return self.main_config.get_config(key, default)

    def set_config(self, key: str, value: str):
        self.main_config.set_config(key, value)

# Create main configuration manager instance
main_config_manager = MainConfigurationManager()

# Example usage:
if __name__ == '__main__':
    main_config_manager.set_config('log_level', 'DEBUG')
    logger = Logger(LogLevel.DEBUG)
    logger.debug('This is a debug message.')
    logger.info('This is an info message.')
    logger.warning('This is a warning message.')
    logger.error('This is an error message.')