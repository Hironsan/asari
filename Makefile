lint:
	poetry run task flake8
	poetry run task black
	poetry run task isort
	poetry run task mypy

test:
	poetry run task test
