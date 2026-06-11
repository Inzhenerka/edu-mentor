from phoenix.client.experiments import run_experiment
from loguru import logger

from evals.harness import prepare_environment, get_user_id, make_retrieval_task
from evals.phoenix_utils import get_phoenix_client, ensure_dataset
from evals.dataset import RetrievalDataset
from evals.metrics.hit import hit
from evals.metrics.recall import recall
from evals.metrics.precision import precision
from evals.metrics.reciprocal_rank import reciprocal_rank
from evals.metrics.correct_rejection import correct_rejection


def main() -> None:
    # Готовим окружение и создаем клиент Phoenix
    config = prepare_environment()
    client = get_phoenix_client(config.observability.endpoint)

    # Загружаем эталонный датасет из файла и передаем его в Phoenix
    dataset = RetrievalDataset.from_yaml_file()
    phoenix_dataset = ensure_dataset(
        client,
        name=f"retrieval_{dataset.version}_{get_user_id()}",
        description="edu-mentor: retrieval",
        examples=dataset.to_example_dicts()
    )

    # Создаем тестовую задачу для вызова ретривера
    task = make_retrieval_task(config)

    # Запускаем эксперимент, собранный из подготовленных компонентов
    exp_result = run_experiment(
        client=client,
        dataset=phoenix_dataset,
        task=task,
        evaluators=[hit, recall, precision, reciprocal_rank, correct_rejection],
        experiment_name=f"retrieval-k{config.rag.retriever.k}-th{config.rag.retriever.score_threshold}",
        experiment_description="Calculated retrieval metrics",
    )
    logger.info(f'Dataset ID: {exp_result["dataset_id"]}  Experiment ID: {exp_result["experiment_id"]}')


if __name__ == "__main__":
    main()
