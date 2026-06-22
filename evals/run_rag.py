from phoenix.client.experiments import run_experiment

from evals.harness import prepare_environment, make_agent_task, get_user_id
from evals.judges.faithfulness import make_faithfulness_judge
from evals.judges.document_relevance import make_document_relevance_judge
from evals.judges.refusal import make_refusal_judge
from evals.dataset import RagDataset
from evals.phoenix_utils import ensure_dataset, get_llm, get_phoenix_client


def main() -> None:
    # Готовим окружение и создаем клиент Phoenix
    config = prepare_environment()
    client = get_phoenix_client(config.observability.endpoint)

    # Загружаем датасет вопросов из файла и передаем его в Phoenix
    dataset = RagDataset.from_yaml_file()
    phoenix_dataset = ensure_dataset(
        client=client,
        name=f"rag_{dataset.version}_{get_user_id()}",
        examples=dataset.to_example_dicts(),
    )

    # Создаем тестовую задачу для вызова агента
    task = make_agent_task(config)

    # Создаем LLM-судью
    judge_llm = get_llm(config.llms["judge"])

    # Запускаем эксперимент с триадой судей качества генерации
    run_experiment(
        client=client,
        dataset=phoenix_dataset,
        task=task,
        evaluators=[
            make_faithfulness_judge(judge_llm),
            make_document_relevance_judge(judge_llm),
            make_refusal_judge(judge_llm),
        ],
        experiment_name=f"rag-k{config.rag.retriever.k}-th{config.rag.retriever.score_threshold}",
        experiment_description="RAG answer quality on the IT mentor dataset",
    )


if __name__ == "__main__":
    main()
