from typing import Any

from phoenix.evals import create_evaluator, Score

from evals.contracts import RetrievalOutput


@create_evaluator(name="precision")
def precision(output: dict[str, Any], expected: dict[str, Any]) -> Score:
    """Точность: сколько выданных чанков оказались эталонными, шум ретривера."""

    # Парсим ожидаемый выход ретривера из эталонного примера
    expected_ids = RetrievalOutput.model_validate(expected).chunk_ids
    # Метрика не имеет смысла, если мы ожидаем пустой список чанков
    if not expected_ids:
        return Score(label="not-applicable", explanation="No expected chunks")

    # Парсим реальный выход ретривера той же моделью
    output_ids = RetrievalOutput.model_validate(output).chunk_ids

    # Защита от деления на ноль. Если ничего не найдено, точность нулевая
    if not output_ids:
        return Score(score=0.0, explanation="No chunks retrieved")

    # Считаем, сколько из ожидаемых чанков ретривер действительно нашёл
    relevant_count = len(set(output_ids) & set(expected_ids))

    # Возвращаем долю релевантных среди всех выданных
    return Score(
        score=relevant_count / len(output_ids),  # Отношение релевантных ко всем выданным
        explanation=f"{relevant_count}/{len(output_ids)} retrieved chunks relevant",
    )
