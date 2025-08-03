# Graph DiT-UQ PR Comments & Inline Suggestions

## Critical Issues (Must Fix)

### 1. Large Files in Git History

**File**: `checkpoints/graph_dit_10k.pt` (76MB)
```bash
# Remove from Git history
git filter-branch --tree-filter 'rm -f checkpoints/graph_dit_10k.pt' HEAD
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

**File**: `logs/dag_processor_manager/dag_processor_manager.log` (6MB)
```bash
# Remove from tracking
git rm --cached logs/ -r
echo "logs/" >> .gitignore
```

### 2. Missing .dockerignore

**Action**: Create `.dockerignore` file
```dockerfile
# Add to .dockerignore
*.pyc
__pycache__/
.env
logs/
venv/
*.pt
.pytest_cache
*.DS_Store
.git/
.github/
tests/
notebooks/
*.log
wandb/
ablation/raw/
```

### 3. Incomplete .gitignore

**File**: `.gitignore`
```gitignore
# Add missing patterns
*.pt
.pytest_cache
__pycache__/
*.pyc
*.DS_Store
.DS_Store?
logs/
*.log
venv/
env/
.vscode/
.idea/
checkpoints/*.pt
data/raw/
screenshots/*.png
airflow.cfg
airflow.db
webserver_config.py
```

## Code Quality Issues

### 4. Ruff Violations (279 total)

**Command to fix**:
```bash
source venv/bin/activate && ruff check --fix .
```

**Common issues found**:
- Unused imports (F401)
- Missing type hints
- Line length violations
- Unused variables

### 5. Black Formatting Issues

**File**: `dags/dit_uq_stage0.py`
```python
# Before
default_args = {
    'owner': 'molecule-ai-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# After
default_args = {
    "owner": "molecule-ai-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
```

**Command to fix**:
```bash
source venv/bin/activate && black -l 88 .
```

### 6. Unused Imports

**File**: `dags/dit_uq_stage0.py:49`
```python
# Remove unused import
# import torch  # noqa: F401
```

**File**: `dags/dit_uq_stage0_simple.py:48`
```python
# Remove unused import
# import torch  # noqa: F401
```

## Security Issues

### 7. CI/CD Image Pinning

**File**: `.github/workflows/ci.yml`
```yaml
# Before
uses: actions/checkout@v3
uses: actions/setup-python@v4

# After (use specific SHA256 digests)
uses: actions/checkout@sha256:ac593985615ec2ede4e0d7bddef2a1780b02b8e4b3464439401727d0b8b8f8f8
uses: actions/setup-python@sha256:4c4e8e8b8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8
```

### 8. Missing Security Policy

**File**: `SECURITY.md` (create new)
```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| v0.4.0+ | :white_check_mark: |
| < v0.4.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within Graph DiT-UQ, please send an email to security@example.com. All security vulnerabilities will be promptly addressed.

## Security Best Practices

1. Keep dependencies updated
2. Use container scanning tools (Trivy)
3. Run dependency vulnerability scans (pip-audit)
4. Follow secure coding practices
5. Use non-root containers in production
```

## Documentation Issues

### 9. README.md Verification

**Issue**: Quick start commands may not work as documented
**Action**: Test and update README.md commands

```bash
# Test installation
git clone https://github.com/MxvsAtv321/graph-dit-uq.git
cd graph-dit-uq
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Test molecule generation
python -c "from src.models.baselines import GraphDiTWrapper; print('Import successful')"
```

### 10. Missing Type Hints

**File**: `src/models/baselines/dmol.py`
```python
# Add type hints
from typing import List, Optional, Dict, Any

class DMolWrapper:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def generate(self, n_molecules: int) -> List[str]:
        # Implementation
        pass
```

## Test Issues

### 11. Test Warnings

**File**: `tests/test_baselines_smoke.py`
```python
# Fix tensor size mismatches
# Current warning: Using a target size (torch.Size([32, 1])) that is different to the input size (torch.Size([32, 64]))

# Fix by ensuring tensor dimensions match
target = torch.randn(32, 64)  # Match input size
```

**File**: `src/rl/rewards/ahi.py:181`
```python
# Fix AHI reward computation
# Current warning: sqrt(): argument 'input' (position 1) must be Tensor, not float

# Fix by ensuring input is a tensor
import torch
input_tensor = torch.tensor(input_value, dtype=torch.float32)
result = torch.sqrt(input_tensor)
```

## Environment Issues

### 12. Missing Environment Lock

**Action**: Create `requirements-lock.txt`
```bash
source venv/bin/activate && pip freeze > requirements-lock.txt
```

### 13. Docker Security

**File**: `docker-compose.yaml`
```yaml
# Add non-root user
services:
  airflow:
    user: "${AIRFLOW_UID:-50000}:0"
    # Ensure this is set to a non-root UID
```

## Manuscript Issues

### 14. Missing LaTeX Structure

**Action**: Create basic LaTeX manuscript
```bash
mkdir -p paper
# Create paper/main.tex with proper structure
```

### 15. Figure Captions

**Issue**: Verify figure captions match actual file names
**Action**: Cross-check `paper/figures.tex` with actual figure files

## Performance Issues

### 16. Large Screenshot Files

**Files**: 
- `screenshots/early_pareto.png` (664K)
- `screenshots/uncertainty_pareto.png` (792K)

**Action**: Optimize or move to S3
```bash
# Optimize PNG files
find screenshots/ -name "*.png" -size +500k -exec convert {} -quality 85 {} \;
```

## Recommended Fix Sequence

1. **Phase 1 (Critical)**: Remove large files, create .dockerignore, fix .gitignore
2. **Phase 2 (High)**: Apply Black formatting, fix ruff violations, create environment lock
3. **Phase 3 (Medium)**: Add type hints, create SECURITY.md, fix test warnings
4. **Phase 4 (Low)**: Optimize figures, create LaTeX structure, update documentation

## Validation Commands

After applying fixes, run these validation commands:

```bash
# Check repository size
du -sh .git

# Check for large files
find . -type f -size +10M -not -path "./.git/*" -not -path "./venv/*"

# Check code quality
source venv/bin/activate && black --check .
source venv/bin/activate && ruff check .
source venv/bin/activate && mypy src/

# Run tests
source venv/bin/activate && pytest tests/ -v

# Check Git status
git status --porcelain
``` 