# Create virtual environment
venv:
	python3 -m venv venv

# Install dependencies
install:
	. venv/bin/activate && pip install -r requirements.txt

# Run the application
run: 
	. venv/bin/activate && flask run --host=localhost --port=3000

# Clean up the virtual environment
clean:
	rm -rf venv

# Command to recreate the virtual environment, install dependencies, and run the app
setup:
	make venv && make install && make run
