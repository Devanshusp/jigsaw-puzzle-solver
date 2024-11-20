# Jigsaw Puzzle Solver

## Environment

```bash
# 1. Create a virtual environment named .venv.
pip install uv
uv .venv
python -m venv .venv # alternatively without uv

# 2. Activate the virtual environment.
source .venv/bin/activate    # macOS and Linux
.venv\Scripts\activate       # Windows

# 3. Install dependencies.
uv pip install -r requirements.txt
pip install -r requirements.txt # alternatively without uv

# 4. Save any additional dependencies to requirements.txt.
uv pip freeze > requirements.txt
pip freeze > requirements.txt # alternatively without uv
```
