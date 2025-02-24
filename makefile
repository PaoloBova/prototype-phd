# Define variables
ENV_NAME = model-ai.prototype-phd
PYTHON_VERSION = 3.10
REQUIREMENTS_FILE = requirements.txt

# Target to create mamba environment
env:
	mamba create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	@echo "Run 'mamba activate $(ENV_NAME)' to activate the environment."

# Install dependencies
deps: 
	mamba run -n $(ENV_NAME) pip install -r $(REQUIREMENTS_FILE)

# Run Jupyter Lab within the environment
lab:
	mamba run -n $(ENV_NAME) jupyter lab

# Clean environment
clean:
	mamba env remove -n $(ENV_NAME)
