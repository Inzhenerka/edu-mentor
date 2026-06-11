from typing import Any

from phoenix.evals import create_evaluator, Score

from evals.contracts import RetrievalOutput


@create_evaluator(name="correct_rejection")
def correct_rejection(output: dict[str, Any], expected: dict[str, Any]) -> Score:
    """Верный отказ: ретривер корректно не вернул ни одного чанка по стороннему запросу.
    Проверка на ложные срабатывания."""

    # Парсим ожидаемый выход ретривера из эталонного примера
    expected_ids = RetrievalOutput.model_validate(expected).chunk_ids
    # "Негативная" метрика: не имеет смысла, если ожидаемый список чанков не пуст
    if expected_ids:
        return Score(label="not-applicable", explanation="Expected chunks present")

    # Парсим реальный выход ретривера той же моделью
    output_ids = RetrievalOutput.model_validate(output).chunk_ids

    # Правильное поведение: на вопрос вне корпуса ретривер ничего не вернул
    if not output_ids:
        return Score(
            score=1,
            label="correct-reject",
            explanation="No chunks retrieved",
        )

    # Ретривер "протёк" - вернул чанки, когда ожидалась пустая выдача
    return Score(
        score=0,
        label="leaked-noise",
        explanation=f"{len(output_ids)} unexpected chunks retrieved",
    )
