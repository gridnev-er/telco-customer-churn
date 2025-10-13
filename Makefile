# Makefile — базовые якоря
.PHONY: setup lint format test app data clean

PY := poetry run

setup:
	poetry install
	pre-commit install
	@echo "✔️  Environment is ready"

lint:
	$(PY) ruff check src tests

format:
	$(PY) ruff format src tests

test:
	$(PY) pytest -q

app:
	$(PY) streamlit run app/main.py

# Пример «пустой» цели для данных — позже привяжем скрипт
data:
	$(PY) python scripts/make_clean.py

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__
