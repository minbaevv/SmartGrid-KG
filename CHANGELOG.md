# Изменения

## 0.3.0 — 2026-10-11
- Контейнеризация (этап 3): `Dockerfile` на python:3.11-slim, uvicorn на `${PORT:-8000}`, HEALTHCHECK на /healthcheck.
- `docker-compose.yml`: сервис api, порт 8000:8000, volumes для `./data` и `./app/ml/model.pkl`.
- `.dockerignore`: venv/, .git/, *.db, .env и кеши не попадают в образ.
- Проверено: все зависимости резолвятся в manylinux-wheel для CPython 3.11 (компиляторы не нужны), YAML валиден.
- GitHub Actions CI (`.github/workflows/ci.yml`): push и PR в main, матрица Python 3.11 и 3.13, установка из requirements.txt, запуск `python -m unittest discover -s tests -v`.
- Первый запуск CI на GitHub (run #1, push ccb7405): обе задачи матрицы — success.
- Проверено в чистом venv Python 3.11.9: установка из requirements.txt (exit 0, без сборки из исходников) и 24/24 теста; на Python 3.13 — 24/24. Флаг `-s tests` обязателен: без него discover находит 0 тестов.
- Уточнено: requirements.txt в образе резолвит scikit-learn 1.9.1 для CPython 3.11, то есть версия совпадает с метаданными локальной модели; прежнее утверждение про 1.7.2 было артефактом слишком строгого тега manylinux2014.
- Сборка `docker build` локально не выполнена: Docker не установлен на рабочей машине.

## 0.2.0 — 2026-10-10
- Асинхронный SQLAlchemy: SQLite/aiosqlite, PostgreSQL/asyncpg.
- Substation, SubstationLoad, snapshot-признаки и маркировка синтетики.
- Идемпотентный трёхмесячный генератор для 8 вымышленных подстанций.
- RandomForest, временные блоки обучения/калибровки/теста, базовая модель.
- Demo-/api/v1/predict: районы, 1–48 будущих часов, актуальная погода, предупреждения.
- 24 теста и реальная проверка прогнозного API; обновлены Markdown-исходники и план.

## 0.1.0 — 2026-10-10
Создан локальный прототип первого этапа: FastAPI, настройки, погода, SQLite, импорт нормализованного CSV, валидация сезонности.
