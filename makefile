# Define commands
.PHONY: install run clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run the application
run:
	python3 app.py
