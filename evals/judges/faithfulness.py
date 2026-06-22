from phoenix.evals import bind_evaluator, LLM
from phoenix.evals.evaluators import Evaluator
from phoenix.evals.metrics.faithfulness import FaithfulnessEvaluator

from evals.contracts import AgentOutput, ExampleInput


def make_faithfulness_judge(llm: LLM) -> Evaluator:
    """Создаем оценщика-судью для проверки верности RAG-ответа источникам."""
    return bind_evaluator(
        FaithfulnessEvaluator(llm=llm),
        input_mapping={
            "input": lambda x: ExampleInput.model_validate(x["input"]).question,
            "output": lambda x: AgentOutput.model_validate(x["output"]).answer,
            "context": lambda x: AgentOutput.model_validate(x["output"]).retrieved_context,
        },
    )
