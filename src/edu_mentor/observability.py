from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from edu_mentor.config import ObservabilityConfig


def setup_observability(config: ObservabilityConfig) -> None:
    """Подключает Phoenix-трассировку для всех вызовов LangChain-агента."""
    tracer_provider = register(
        project_name=config.project_name,
        endpoint=config.endpoint,
        batch=True,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
