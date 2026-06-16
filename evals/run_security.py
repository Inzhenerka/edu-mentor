from phoenix.client.experiments import run_experiment

from evals.harness import prepare_environment, make_agent_task, get_user_id
from evals.judges.security import make_security_judge
from evals.dataset import SecurityDataset
from evals.phoenix_utils import ensure_dataset, get_llm, get_phoenix_client


def main() -> None:
    # Готовим окружение и создаем клиент Phoenix
    config = prepare_environment()
    client = get_phoenix_client(config.observability.endpoint)

    # Загружаем эталонный датасет из файла и передаем его в Phoenix
    dataset = SecurityDataset.from_yaml_file()
    phoenix_dataset = ensure_dataset(
        client=client,
        name=f"security_{dataset.version}_{get_user_id()}",
        examples=dataset.to_example_dicts(),
    )

    # Создаем тестовую задачу для вызова агента
    task = make_agent_task(config)

    # Создаем LLM-судью
    judge_llm = get_llm(config.llms["judge"])

    # Запускаем эксперимент, собранный из подготовленных компонентов
    run_experiment(
        client=client,
        dataset=phoenix_dataset,
        task=task,
        evaluators=[make_security_judge(judge_llm)],  # создаем оценщика-судью
        experiment_name="security",
        experiment_description="Prompt-injection and jailbreak resistance",
    )


if __name__ == "__main__":
    main()
