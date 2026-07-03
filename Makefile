# Convenience commands. Run `make help` to list them.
.PHONY: help setup data train evaluate predict test lint format app tensorboard clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create a virtualenv and install dependencies
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:  ## Download the MIT-BIH Arrhythmia Database
	python download_data.py

train:  ## Train the model
	python -m src.train --config config.yaml

evaluate:  ## Evaluate the best checkpoint on the test set
	python -m src.evaluate --checkpoint models/best_model.pt

predict:  ## Predict all beats in record 100 (RECORD=xxx to change)
	python -m src.predict --record $(or $(RECORD),100) --checkpoint models/best_model.pt

test:  ## Run the unit test suite
	pytest -q

lint:  ## Lint with Ruff
	ruff check src tests

format:  ## Auto-format with Ruff
	ruff format src tests

app:  ## Launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

tensorboard:  ## Open TensorBoard on the training logs
	tensorboard --logdir outputs/tensorboard

clean:  ## Remove caches and generated artefacts (keeps raw data)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ outputs/tensorboard
	rm -f models/*.pt outputs/figures/*.png outputs/confusion_matrix/*.png outputs/reports/*
