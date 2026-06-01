from dotenv import load_dotenv

from edu_mentor.config import Config
from edu_mentor.observability import setup_observability


def prepare_environment() -> Config:
    """Загрузка конфигурации приложения, переменных окружения и инструментация."""
    load_dotenv()
    config = Config.from_yaml_file("config.yml")
    setup_observability(config.observability)
    return config
