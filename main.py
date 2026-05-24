from fastapi.testclient import TestClient

from edu_tutor.api import app, TutorResponse

PROMPTS = [
    # Вызов RAG
    "Как посмотреть последние коммиты в Git?",
    # Вызов инструмента exercise
    "Спасибо! Дай задачу по гиту, что-то начального уровня",
    # Вызов инструмента consultation
    "Давай я лучше созвонюсь с куратором по этой теме завтра в два часа",
]

with TestClient(app) as client:
    for prompt in PROMPTS:
        print(f"\n👤: {prompt}")
        api_response = client.post(
            "/ask",
            data={"question": prompt, "thread_id": "1"},
        )
        response = TutorResponse.model_validate(api_response.json())
        print(f"🤖: {response.content}")
        if response.tool_calls:
            print(f"  🛠️ Инструменты: {', '.join(response.tool_calls)}")
        if response.sources:
            print("  📚 Источники:")
            for ref_num, meta in response.sources.items():
                print(f"   [{ref_num}] «{meta.title}», {meta.author} - {meta.chunk_id}")
