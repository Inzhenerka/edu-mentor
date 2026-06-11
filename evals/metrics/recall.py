from typing import Any

from phoenix.evals import create_evaluator, Score

from evals.contracts import RetrievalOutput


@create_evaluator(name="recall")
def recall(output: dict[str, Any], expected: dict[str, Any]) -> Score:
    """Полнота: сколько эталонных чанков ретривер нашел, покрытие gold set."""

    # Парсим ожидаемый выход ретривера из эталонного примера
    expected_ids = RetrievalOutput.model_validate(expected).chunk_ids
    # Метрика не имеет смысла, если мы ожидаем пустой список чанков
    if not expected_ids:
        return Score(label="not-applicable", explanation="No expected chunks")

    # Парсим реальный выход ретривера той же моделью
    output_ids = RetrievalOutput.model_validate(output).chunk_ids

    # Считаем, сколько из ожидаемых чанков ретривер действительно нашёл
    relevant_count = len(set(output_ids) & set(expected_ids))

    # Возвращаем долю найденных от всех ожидаемых
    return Score(
        score=relevant_count / len(expected_ids),  # Отношение найденных ко всем ожидаемым
        explanation=f"{relevant_count}/{len(expected_ids)} expected chunks found",
    )
