from phoenix.evals import bind_evaluator, LLM
from phoenix.evals.evaluators import Evaluator
from phoenix.evals.metrics.refusal import RefusalEvaluator

from evals.contracts import AgentOutput, ExampleInput


def make_refusal_judge(llm: LLM) -> Evaluator:
    """Создаем судью-оценщика отказов агента."""
    return bind_evaluator(
        RefusalEvaluator(llm=llm),
        input_mapping={
            "input": lambda x: ExampleInput.model_validate(x["input"]).question,
            "output": lambda x: AgentOutput.model_validate(x["output"]).answer,
        },
    )
