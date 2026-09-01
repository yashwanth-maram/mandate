.PHONY: gen eval demo test api

gen:
	uv run python -m bench.generator --seed 42 --n 500

eval:
	uv run python -m eval.runner --seed 42

demo:
	uv run python -m demo.scenarios

test:
	uv run pytest -q

api:
	uv run uvicorn api.main:app --reload
