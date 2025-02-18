# prototype-phd

A library where I communicate the key tools and insights I develop during my PhD (with an emphasis on AI Governance).

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Running the Notebook](#running-the-notebook)
- [Makefile Targets](#makefile-targets)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains the `prototype-phd` library, which includes various tools and models developed during my PhD research. The focus is on AI Governance, and this repository includes a notebook that analyzes the AI Trust model described in the
following preprint: https://arxiv.org/abs/2403.09510.

## Setup

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/prototype-phd.git
    cd prototype-phd
    ```

2. **Create the environment**:
    Use the provided Makefile to create a mamba environment with the specified Python version.
    ```bash
    make env
    ```

3. **Activate the environment**:
    ```bash
    mamba activate model-ai.phd_prototype
    ```

4. **Install dependencies**:
    ```bash
    make deps
    ```

## Running the Notebook

To run the Jupyter notebook and plot the results of the AI Trust model, follow these steps:

1. **Start Jupyter Lab**:
    ```bash
    make lab
    ```

2. **Open the notebook**:
    In Jupyter Lab, navigate to the `notebooks` directory and open a notebook.

3. **Run the notebook**:
    Execute the cells in the notebook to perform the analysis and plot the results.

## Makefile Targets

The Makefile includes several targets to help with setup and running the project:

- `env`: Creates the mamba environment with the specified Python version.
- `deps`: Installs the dependencies listed in `requirements.txt`.
- `lab`: Runs Jupyter Lab within the environment.
- `clean`: Removes the mamba environment.

Example usage:
```bash
make env       # Create the environment
make deps      # Install dependencies
make lab       # Run Jupyter Lab
make clean     # Clean the environment
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

