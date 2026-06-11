from typing import Any

from phoenix.evals import create_evaluator, Score

from evals.contracts import RetrievalOutput


@create_evaluator(name="reciprocal_rank")
def reciprocal_rank(output: dict[str, Any], expected: dict[str, Any]) -> Score:
    """Обратный ранг: позиция первого правильного чанка, оценка порядка выдачи."""

    # Парсим ожидаемый выход ретривера из эталонного примера
    expected_ids = RetrievalOutput.model_validate(expected).chunk_ids
    # Метрика не имеет смысла, если мы ожидаем пустой список чанков
    if not expected_ids:
        return Score(label="not-applicable", explanation="No expected chunks")

    # Парсим реальный выход ретривера той же моделью
    output_ids = RetrievalOutput.model_validate(output).chunk_ids

    # Идём по выдаче начиная с первого чанка
    for rank, chunk_id in enumerate(output_ids, start=1):
        if chunk_id in expected_ids:
            # Встретили первый эталонный чанк в ответе ретривера - считаем обратный ранг
            return Score(
                score=1.0 / rank,  # Обратная величина к позиции чанка
                explanation=f"First match at rank {rank}",
            )

    # Во всей выдаче не встретилось ни одного релевантного чанка
    return Score(score=0.0, explanation=f"No match in top-{len(output_ids)}")
