from datetime import datetime

from langchain.tools import tool
from openinference.instrumentation.langchain import get_current_span


@tool
def book_consultation(slot: str, topic: str) -> str:
    """Бронирует слот для встречи с куратором-человеком.

    Используй ТОЛЬКО когда студент явно подтвердил конкретное время в ISO-формате
    (например, «2026-05-15T15:00»). Если время нечёткое («как-нибудь», «вечером»,
    «на этой неделе»), сначала переспроси у студента точное время, а затем уже вызывай тул.

    Аргументы:
        slot: дата и время в ISO-формате 'YYYY-MM-DDTHH:MM'.
        topic: краткая формулировка темы консультации.
    """
    try:
        parsed_slot = datetime.fromisoformat(slot.strip())
    except ValueError:
        return (
            f"TOOL_REJECTED: slot должен быть в ISO-формате 'YYYY-MM-DDTHH:MM'. "
            f"Получено: {slot!r}. Это служебное сообщение тебе как агенту, "
            f"пользователю его не пересказывай."
        )

    if not topic or len(topic.strip()) < 3:
        return (
            f"TOOL_REJECTED: topic должен содержать осмысленную тему "
            f"консультации (минимум 3 символа). Получено: {topic!r}. "
            f"Это служебное сообщение тебе как агенту, пользователю его "
            f"не пересказывай."
        )

    # Мок-бронирование - в продакшене здесь был бы вызов внутреннего API расписания или обращение к БД.
    span = get_current_span()  # Получаем доступ к спану инструмента
    span.add_event("slot_booked", attributes={
        "topic": topic, "slot": parsed_slot.isoformat()
    })  # Создаем событие в Phoenix для отслеживания
    return (
        f"Слот забронирован: {parsed_slot.strftime('%d.%m.%Y в %H:%M')}, "
        f"тема: «{topic.strip()}». Куратор получит уведомление о встрече."
    )
