from typing import Any

from phoenix.evals import create_evaluator, Score

from evals.contracts import RetrievalOutput


@create_evaluator(name="hit")
def hit(output: dict[str, Any], expected: dict[str, Any]) -> Score:
    """Оценивает, есть ли среди найденных документов хотя бы один релевантный."""

    # Парсим ожидаемый выход ретривера из эталонного примера
    expected_ids = RetrievalOutput.model_validate(expected).chunk_ids
    # Метрика не имеет смысла, если мы ожидаем пустой список чанков
    if not expected_ids:
        return Score(label="not-applicable", explanation="No expected chunks")

    # Парсим реальный выход ретривера той же моделью
    output_ids = RetrievalOutput.model_validate(output).chunk_ids

    # Ищем количество совпавших чанков
    matched_count = len(set(output_ids) & set(expected_ids))

    # Возвращаем не просто флаг, а оценку с пояснениями
    return Score(
        score=int(matched_count > 0),  # True, если есть совпадения
        label="hit" if matched_count else "miss",  # Более наглядная метка для UI
        explanation=f"{matched_count}/{len(expected_ids)} expected chunks found",  # Объяснение для UI
    )
