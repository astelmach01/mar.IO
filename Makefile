# Makefile to manage Python dependencies using pip-compile and pip
all: check

# The input requirements.in file
REQUIREMENTS_IN = requirements.in

# The generated requirements.txt file
REQUIREMENTS_TXT = requirements.txt

.PHONY: compile install push format type check

format:
	ruff format .

type:
	mypy .

check: format type

compile: $(REQUIREMENTS_TXT)

install: compile
	pip install -r $(REQUIREMENTS_TXT)


$(REQUIREMENTS_TXT): $(REQUIREMENTS_IN)
	pip-compile $(REQUIREMENTS_IN) -o $(REQUIREMENTS_TXT)

push: check
	@if [ -z "$(message)" ]; then \
		echo "Please specify a commit message: make commit message='Your message here'"; \
		exit 1; \
	fi
	git add .
	git commit -m "$(message)"
	git push
