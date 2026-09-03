.PHONY: gen eval demo test api sweep

gen:
	uv run python -m bench.generator --seed 42 --n 500

eval:
	uv run python -m eval.runner --seed 42 --model google/gemini-2.5-flash --cache-only

demo:
	uv run python -m demo.scenarios

test:
	uv run pytest -q

api:
	uv run uvicorn api.main:app --reload

sweep:
	uv run python -m eval.sweep --seed 42 --model google/gemini-2.5-flash
