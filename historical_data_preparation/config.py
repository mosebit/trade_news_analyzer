"""
Configuration loader for Trade News Analyzer.
Loads configuration from YAML file and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    path: str
    collection_name: str
    embedding_model: str
    similarity_metric: str


@dataclass
class SimilaritySearchConfig:
    vector_top_k: int
    bm25_enabled: bool
    bm25_top_k: int
    bm25_weight: float
    reranking_enabled: bool
    reranking_model: str
    reranking_top_k: int
    duplicate_threshold: float


@dataclass
class LLMConfig:
    use_custom_client: bool
    model: str
    api_key: str
    base_url: str
    temperature_enrichment: float
    temperature_duplicate_check: float
    max_retries: int
    retry_delay_seconds: int


@dataclass
class TickerInfo:
    ticker: str
    description: str


@dataclass
class SmartLabConfig:
    base_url: str
    request_delay_seconds: float
    max_retries: int
    retry_delay_seconds: int
    timeout_seconds: int
    headers: Dict[str, str]


@dataclass
class LoggingConfig:
    level: str
    format: str
    log_to_file: bool
    log_file_path: str
    rotation: str
    retention: str


@dataclass
class PipelineConfig:
    target_date: str
    save_stats_interval: int
    continue_on_error: bool


class Config:
    """Main configuration class."""

    def __init__(self, config_path: Optional[str] = None):
        # Load environment variables first
        load_dotenv()

        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        # Load YAML config
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # Replace environment variable placeholders
        self._replace_env_vars(self._config)

        # Parse into dataclasses
        self.database = DatabaseConfig(**self._config['database'])
        self.similarity_search = SimilaritySearchConfig(**self._config['similarity_search'])
        self.llm = LLMConfig(**self._config['llm'])
        self.tickers = [TickerInfo(**t) for t in self._config['tickers']]
        self.smartlab = SmartLabConfig(**self._config['smartlab'])
        self.logging = LoggingConfig(**self._config['logging'])
        self.pipeline = PipelineConfig(**self._config['pipeline'])

    def _replace_env_vars(self, config: Any) -> None:
        """Recursively replace ${VAR_NAME} with environment variables."""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var, "")

                    # Special handling for boolean values
                    if key == "use_custom_client":
                        config[key] = env_value.lower() in ("true", "1", "yes")
                    else:
                        config[key] = env_value
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(config, list):
            for item in config:
                self._replace_env_vars(item)

    def get_tickers_dict(self) -> Dict[str, str]:
        """Get tickers as a dictionary {ticker: description}."""
        return {t.ticker: t.description for t in self.tickers}

    def get_ticker_list(self) -> List[str]:
        """Get list of ticker symbols."""
        return [t.ticker for t in self.tickers]


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = Config(config_path)
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Database path: {config.database.path}")
    print(f"Embedding model: {config.database.embedding_model}")
    print(f"Tickers: {config.get_ticker_list()}")
    print(f"LLM model: {config.llm.model}")
    print(f"Reranking enabled: {config.similarity_search.reranking_enabled}")
