# Define the requirements file
REQUIREMENTS_FILE = requirements.txt

# Generate requirements.txt from installed packages
freeze:
	pip freeze > $(REQUIREMENTS_FILE)

# Install dependencies from requirements.txt
install:
	pip install -r $(REQUIREMENTS_FILE)
