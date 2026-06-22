from phoenix.evals import bind_evaluator, LLM
from phoenix.evals.evaluators import Evaluator
from phoenix.evals.metrics.document_relevance import DocumentRelevanceEvaluator

from evals.contracts import AgentOutput, ExampleInput


def make_document_relevance_judge(llm: LLM) -> Evaluator:
    """Создаем судью-оценщика релевантности найденного контекста вопросу."""
    return bind_evaluator(
        DocumentRelevanceEvaluator(llm=llm),
        input_mapping={
            "input": lambda x: ExampleInput.model_validate(x["input"]).question,
            "document_text": lambda x: AgentOutput.model_validate(x["output"]).retrieved_context,
        },
    )
