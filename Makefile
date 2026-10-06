.PHONY: env smoke report validate

env:
	python scripts/check_env.py

smoke:
	python scripts/smoke_test.py

report:
	bash scripts/build_report.sh

validate:
	python scripts/validate_repo.py
