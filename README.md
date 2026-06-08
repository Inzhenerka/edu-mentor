# edu-mentor

**Фортран** - IT-ментор для студентов, изучающих программирование.

Содержит:

- **RAG-обвязку**: подключение к удалённой коллекции Qdrant с уже проиндексированным корпусом (инжестинг выполняется в [qdrant-stand](https://github.com/Inzhenerka/qdrant-stand))
- **Два тула** (`suggest_exercise` - генеративный; `book_consultation` - safety-чувствительный, требует явный ISO-слот) с типизированными аргументами через `@tool`

## Подготовка

```bash
uv sync

cp .env.example .env
# отредактируйте .env, заполните значения переменных
```

## Запуск

```bash
# REST API + браузерный демо-чат на http://localhost:8000/demo
uv run fastapi dev
```

## Оценка качества (evals)

Запуск оценки ретривера:

```bash
uv run python -m evals.run_retrieval
```
