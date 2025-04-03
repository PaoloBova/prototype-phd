# Python Coding Patterns for Scientific Computing Projects

This document outlines a set of coding patterns designed to produce robust, maintainable, and type-safe scientific computing projects in Python. We focus on using Pydantic for data validation, leveraging protocols for minimal coupling and flexible interfaces, employing multimethods for dispatch, testing with Hypothesis, and managing dependencies with Poetry.

---

## High-Level Summary

- **Pydantic Models for Input Validation:**  
  Use Pydantic models to validate function inputs and encapsulate configuration or data. This ensures that data passed between functions is well-structured and type-safe.

- **Single-Argument Functions and Destructuring:**  
  Design functions to accept a single Pydantic model. Destructure parameters explicitly using attribute access (e.g., `config.x`) for clarity and to leverage static type checking.

- **Unpacking Pydantic Models with `**`:**  
  Easily create new model instances by unpacking existing ones (e.g., `ProcessedConfig(**config.dict(), result=computed)`), which helps to chain data processing steps.

- **Protocols for Minimal Coupling:**  
  Use Python’s `Protocol` (with `@runtime_checkable`) to specify the minimal interface an object must have. This is useful for writing generic functions that operate on any object with the required properties.

- **Multimethods and Dispatch on Protocols:**  
  Implement multimethods that dispatch based on the type or structure (via protocols) of the passed object. This allows different functions to be executed for different classes or interfaces.

- **Testing with Hypothesis and Pydantic:**  
  Combine Hypothesis with Pydantic to generate valid test inputs automatically using strategies like `st.builds()`, ensuring robust property-based testing.

- **Static Type Checking with mypy:**  
  Enforce type correctness throughout the project with mypy. Configuration can be placed in a separate file (e.g. `mypy.ini`) or directly in the `pyproject.toml` for convenience.

- **Project Management with Poetry:**  
  Use Poetry to manage dependencies and package organization via a `pyproject.toml` file, ensuring consistency between development and production environments.

- **High-Quality Documentation:**  
  Document functions and classes thoroughly. For Pydantic models, document each property with clear descriptions. Functions should detail behavior and, when parameters are Pydantic models already well documented, reference that documentation to avoid redundancy.

- **Advanced Pydantic Configurations:**  
  When needed, you can configure Pydantic models to allow mutation or arbitrary types (using `allow_mutation=True` and `arbitrary_types_allowed=True` in the model's `Config` class). However, these settings should be used sparingly and with caution.

---

## Detailed Explanation and Examples

### 1. Using Pydantic Models for Input Validation and Data Integrity

Pydantic models allow you to define your data schemas and validate input at instantiation. This prevents many runtime errors by ensuring data conforms to the expected structure.

> **Note:** For each property in a Pydantic model, it is recommended to add a detailed description using the `Field` function. This makes the model self-documenting and assists with generating user-friendly API docs.

#### Example: Input and Processed Config Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class InputConfig(BaseModel):
    x: float = Field(..., description="The x-coordinate value.")
    y: float = Field(..., description="The y-coordinate value.")
    description: Optional[str] = Field(None, description="Optional description.")

class ProcessedConfig(InputConfig):
    result: float = Field(..., description="The computed result based on x and y.")

def process_config(config: InputConfig) -> ProcessedConfig:
    computed = config.x * config.y
    # Unpack all fields from config and add 'result'
    return ProcessedConfig(**config.dict(), result=computed)

# Usage:
input_conf = InputConfig(x=3.0, y=4.0)
processed_conf = process_config(input_conf)
print(processed_conf)
```

> **Advanced Configurations:**  
> If needed, you can allow mutation or support arbitrary types in your models by configuring the inner `Config` class:
> 
> ```python
> class AgentModel(BaseModel):
>     position: np.ndarray
>     energy: Optional[float] = None
> 
>     class Config:
>         allow_mutation = True
>         arbitrary_types_allowed = True
> ```
> Use these settings cautiously, as they can bypass some of Pydantic’s guarantees.

---

### 2. Single-Argument Functions and Explicit Destructuring

Each function is designed to accept a single argument (typically a Pydantic model) and then destructure needed parameters explicitly. This ensures clarity and takes advantage of static type checking.

#### Best Practice

```python
def compute_values(config: InputConfig) -> float:
    # Explicitly extract values from the Pydantic model.
    x = config.x
    y = config.y
    return x * y
```

This approach makes the code self-documenting and enables static analysis tools like mypy to catch type issues early.

---

### 3. Unpacking Pydantic Models with `**`

When you need to create a new model from an existing one (perhaps augmenting it with additional fields), using the unpacking operator (`**`) simplifies the code.

#### Example

```python
def process_config(config: InputConfig) -> ProcessedConfig:
    computed = config.x * config.y
    return ProcessedConfig(**config.dict(), result=computed)
```

This pattern scales well even if your models have many properties and allows you to chain processing steps.

---

### 4. Using Protocols for Flexibility and Minimal Coupling

Protocols let you specify the minimal interface required for a function without enforcing a strict inheritance hierarchy. They work especially well for generic low-level functions.

#### Example: Protocol for Objects with a `position`

```python
from typing import Protocol, runtime_checkable, Optional
import numpy as np

@runtime_checkable
class HasPosition(Protocol):
    position: np.ndarray

@runtime_checkable
class HasPositionAndEnergy(Protocol):
    position: np.ndarray
    energy: Optional[float]

class Agent:
    def __init__(self, position: np.ndarray, energy: Optional[float] = None) -> None:
        self.position = position
        self.energy = energy

def update_agent(agent: HasPositionAndEnergy,
                 target: np.ndarray,
                 step: float = 0.1) -> None:
    if not isinstance(agent, HasPositionAndEnergy):
        raise TypeError("The agent must have 'position' and 'energy' attributes.")
    
    if agent.energy is None:
        print("Energy not set. Defaulting energy to 1.0.")
        agent.energy = 1.0

    direction = target - agent.position
    norm = np.linalg.norm(direction)
    if norm > 0:
        agent.position += step * (direction / norm)
    
    print("Updated position:", agent.position, "Energy:", agent.energy)

# Demonstration:
agent1 = Agent(np.array([0.0, 0.0]), energy=None)
target = np.array([1.0, 1.0])
update_agent(agent1, target, step=0.5)
```

Protocols enable functions to accept any object that satisfies the required interface, promoting flexibility.

---

### 5. Multimethods and Dispatch on Protocols

Multimethods allow you to define functions that behave differently based on the type or structure of their input. When combined with runtime-checkable protocols, they offer a powerful way to dispatch on interfaces rather than concrete classes.

#### Multimethod Utilities (Simplified)

```python
def multi(dispatch_fn):
    def _inner(*args, **kwargs):
        key = dispatch_fn(*args, **kwargs)
        fn = _inner.__multi__.get(key, _inner.__multi_default__)
        return fn(*args, **kwargs)
    _inner.__dispatch_fn__ = dispatch_fn
    _inner.__multi__ = {}
    _inner.__multi_default__ = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("Unsupported type"))
    return _inner

def method(dispatch_fn, dispatch_key=None):
    def apply_decorator(fn):
        if dispatch_key is None:
            dispatch_fn.__multi_default__ = fn
        else:
            dispatch_fn.__multi__[dispatch_key] = fn
        return dispatch_fn
    return apply_decorator
```

#### Example: Dispatching on Agent Types

```python
from pydantic import BaseModel
from typing import Optional
import numpy as np

class AgentModel(BaseModel):
    position: np.ndarray
    energy: Optional[float] = None

    class Config:
        allow_mutation = True

def dispatch_agent(agent, *args, **kwargs):
    if isinstance(agent, AgentModel):
        return "pydantic_agent"
    elif isinstance(agent, HasPosition):
        return "generic_agent"
    else:
        return "unsupported"

agent_handler = multi(dispatch_agent)

@method(agent_handler, "pydantic_agent")
def handle_pydantic_agent(agent: AgentModel, target: np.ndarray, step: float):
    print("Handling pydantic agent")
    direction = target - agent.position
    norm = np.linalg.norm(direction)
    if norm > 0:
        agent.position += step * (direction / norm)
    return agent

@method(agent_handler, "generic_agent")
def handle_generic_agent(agent: HasPosition, target: np.ndarray, step: float):
    print("Handling generic agent")
    direction = target - agent.position
    norm = np.linalg.norm(direction)
    if norm > 0:
        agent.position += step * (direction / norm)
    return agent

@method(agent_handler)
def handle_default(agent, *args, **kwargs):
    raise ValueError("Unsupported agent type.")

# Usage:
target = np.array([1.0, 1.0])
agent1 = AgentModel(position=np.array([0.0, 0.0]), energy=10.0)
agent_handler(agent1, target, 0.5)
```

This design enables multimethods to dispatch based on whether an agent is a Pydantic model or simply conforms to a protocol.

---

### 6. Testing with Hypothesis and Pydantic

Hypothesis is used for property-based testing. By combining it with Pydantic, you can generate valid instances of your models automatically.

#### Example Test Using Hypothesis

```python
import numpy as np
import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel
from typing import Optional

class InputConfig(BaseModel):
    x: float
    y: float
    description: Optional[str] = None

class ProcessedConfig(InputConfig):
    result: float

def process_config(config: InputConfig) -> ProcessedConfig:
    computed = config.x * config.y
    return ProcessedConfig(**config.dict(), result=computed)

@given(
    config=st.builds(
        InputConfig,
        x=st.floats(min_value=0.1, max_value=100.0),
        y=st.floats(min_value=0.1, max_value=100.0),
        description=st.one_of(st.none(), st.text(max_size=50))
    )
)
def test_process_config(config: InputConfig):
    result = process_config(config)
    assert np.isclose(result.result, config.x * config.y)

if __name__ == "__main__":
    pytest.main([__file__])
```

Using `st.builds()` generates valid model instances and ensures your processing function works over a broad range of inputs.

---

### 7. Static Type Checking with mypy

mypy helps catch type errors early. Install it via pip and set up a configuration file to enforce strict type checking.

#### Installation

```bash
pip install mypy
```

#### Configuration

You can place mypy configuration either in a `mypy.ini` file or directly in your `pyproject.toml` under the `[tool.mypy]` section. For example, in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.9"
disallow_untyped_defs = true
ignore_missing_imports = true
strict_optional = true
```

Run mypy on your project:

```bash
mypy path/to/your/project
```

Even when using unpacking (`**config.dict()`), if your Pydantic models are well-defined, most type mismatches will be caught either during model instantiation or by mypy when checking function calls.

---

### 8. Organizing Projects with Poetry and pyproject.toml

Poetry simplifies dependency management and packaging. Use a `pyproject.toml` file that conforms to common standards.

#### Example `pyproject.toml`

```toml
[tool.poetry]
name = "my-scientific-project"
version = "0.1.0"
description = "A sample scientific computing project using pydantic, hypothesis, and multimethods."
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
packages = [
    { include = "src" }
]

[tool.poetry.dependencies]
python = "^3.8"
pydantic = "^1.10.0"
numpy = "^1.22"
matplotlib = "^3.5"
hypothesis = "^6.50"

[tool.poetry.dev-dependencies]
pytest = "^7.0"
mypy = "^0.971"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

#### Steps to Get Started

1. **Install Poetry:**  
   ```bash
   pip install poetry
   ```

2. **Initialize the Project:**  
   ```bash
   poetry init
   ```
   Edit the generated `pyproject.toml` as shown above.

3. **Install Dependencies:**  
   ```bash
   poetry install
   ```

4. **Run Tools:**  
   ```bash
   poetry run mypy .
   poetry run pytest
   ```

This setup ensures your project dependencies and configurations are managed consistently across development and production environments.

---

### 9. Documenting Functions and Classes

Maintaining high-quality documentation is essential. Here are some guidelines:

- **Pydantic Models:**  
  - Document each property using `Field(..., description="...")`.  
  - This makes the model self-documenting and improves auto-generated documentation.

- **Functions:**  
  - Include a clear docstring explaining the function's purpose.  
  - Document parameters and return values explicitly, even if some parameters are Pydantic models that are already documented. You can note that detailed documentation is available on the model if needed.
  - Describe side effects (e.g., saving data) or any assumptions made by the function.

- **Classes:**  
  - Document the overall behavior and intended use of the class.  
  - For methods, focus on what each method does rather than rehashing parameter types that are already defined in the class docstring.

Following these practices ensures that your codebase is both user-friendly and maintainable.

---

## Conclusion

By combining these coding patterns:

- **Pydantic models** enforce data integrity and allow easy validation.
- **Explicit destructuring** and **unpacking** keep functions clear and type-safe.
- **Protocols** provide flexibility and minimal coupling.
- **Multimethods** enable dynamic dispatch based on object interfaces.
- **Hypothesis testing** paired with Pydantic ensures robust, property-based tests.
- **mypy** catches type errors early, while **Poetry** manages dependencies and project organization.
- **Thorough documentation** further enhances code clarity and maintainability.

These practices create a robust and maintainable codebase well-suited for complex scientific computing projects.