from evals.harness import prepare_environment
from evals.phoenix_utils import get_phoenix_client

# Загружаем конфигурацию приложения
config = prepare_environment()

# Настраиваем мониторинг и создаем клиент Phoenix
phoenix_client = get_phoenix_client(config.observability.endpoint)

# Тест клиента: получаем список проектов
projects = phoenix_client.projects.list()
print(f"Найдено проектов: {len(projects)}")
