# Create virtual environment
venv:
	python3 -m venv venv

# Install dependencies inside the virtual environment
install:
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Run the application on port 3000
run: 
	. venv/bin/activate && flask run --host=0.0.0.0 --port=3000

# Clean up the virtual environment
clean:
	rm -rf venv

# Full setup to create virtual environment, install dependencies, and run the app
setup:
	make venv && make install && make run

