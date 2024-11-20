# Activate Environment

## Using PIP

```bash
# 1. Create a virtual environment at .venv.
   python -m venv .venv

# 3. Activate the virtual environment.
   source .venv/bin/activate    # macOS and Linux
   .venv\Scripts\activate       # Windows

# 4. Install dependencies.
   pip install -r requirements.txt

# 5. Save any additional dependencies to requirements.txt.
   pip freeze > requirements.txt
```

## Using UV

```bash
# 1. Install the 'uv' package (if not already installed).
   pip install uv

# 3. Create a virtual environment at .venv.
   uv venv

# 4. Activate the virtual environment.
   source .venv/bin/activate    # macOS and Linux
   .venv\Scripts\activate       # Windows

# 5. Install dependencies.
   uv pip install -r requirements.txt

# 5. Save any additional dependencies to requirements.txt.
   uv pip freeze > requirements.txt
```
