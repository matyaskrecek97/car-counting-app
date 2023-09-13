# Define the requirements file
REQUIREMENTS_FILE = requirements.txt

# Generate requirements.txt from installed packages
freeze: activate
	pip freeze > $(REQUIREMENTS_FILE)

# Install dependencies from requirements.txt
install: activate
	pip install -r $(REQUIREMENTS_FILE)
