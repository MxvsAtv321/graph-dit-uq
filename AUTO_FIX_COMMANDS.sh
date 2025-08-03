#!/bin/bash
# Graph DiT-UQ Auto-Fix Commands
# Run these commands to fix the identified issues

set -e  # Exit on any error

echo "🔧 Graph DiT-UQ Auto-Fix Script"
echo "================================="

# Phase 1: Critical Fixes
echo "📋 Phase 1: Critical Fixes"
echo "---------------------------"

# 1. Create .dockerignore
echo "Creating .dockerignore..."
cat > .dockerignore << 'EOF'
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
EOF

# 2. Update .gitignore
echo "Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Build artifacts
*.pt
.pytest_cache
__pycache__/
*.pyc

# System files
*.DS_Store
.DS_Store?

# Logs
logs/
*.log

# Virtual environment
venv/
env/

# IDE
.vscode/
.idea/

# Large files
checkpoints/*.pt
data/raw/
screenshots/*.png

# Airflow
airflow.cfg
airflow.db
webserver_config.py
EOF

# 3. Remove large files from Git tracking
echo "Removing large files from Git tracking..."
git rm --cached logs/ -r || true
git rm --cached checkpoints/graph_dit_10k.pt || true
git rm --cached screenshots/early_pareto.png || true
git rm --cached screenshots/uncertainty_pareto.png || true

# Phase 2: Code Quality Fixes
echo ""
echo "📋 Phase 2: Code Quality Fixes"
echo "------------------------------"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# 4. Apply Black formatting
echo "Applying Black formatting..."
if command -v black &> /dev/null; then
    black -l 88 . || echo "Black not available, skipping..."
else
    echo "Black not found, install with: pip install black"
fi

# 5. Fix ruff violations
echo "Fixing ruff violations..."
if command -v ruff &> /dev/null; then
    ruff check --fix . || echo "Ruff not available, skipping..."
else
    echo "Ruff not found, install with: pip install ruff"
fi

# 6. Create requirements lock file
echo "Creating requirements lock file..."
if command -v pip &> /dev/null; then
    pip freeze > requirements-lock.txt
else
    echo "pip not found, skipping requirements lock..."
fi

# Phase 3: Security & Documentation
echo ""
echo "📋 Phase 3: Security & Documentation"
echo "-----------------------------------"

# 7. Create SECURITY.md
echo "Creating SECURITY.md..."
cat > SECURITY.md << 'EOF'
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

## Known Issues

None at this time.
EOF

# 8. Create basic LaTeX structure
echo "Creating basic LaTeX manuscript structure..."
mkdir -p paper
cat > paper/main.tex << 'EOF'
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Uncertainty-Aware Multi-Objective Molecular Design via Graph Diffusion Transformers with Reinforcement Learning}
\author{Shrirang Shivesh}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
TODO: Add abstract
\end{abstract}

\section{Introduction}
TODO: Add introduction

\section{Methods}
TODO: Add methods

\section{Results}
TODO: Add results

\section{Discussion}
TODO: Add discussion

\section{Conclusion}
TODO: Add conclusion

\bibliographystyle{plain}
\bibliography{references}

\end{document}
EOF

# Phase 4: CI/CD Improvements
echo ""
echo "📋 Phase 4: CI/CD Improvements"
echo "------------------------------"

# 9. Update GitHub Actions to use pinned digests
echo "Updating GitHub Actions to use pinned digests..."
# Note: This requires manual editing of .github/workflows/ci.yml
echo "⚠️  Manual action required: Update .github/workflows/ci.yml to use SHA256 digests"

# Phase 5: Validation
echo ""
echo "📋 Phase 5: Validation"
echo "----------------------"

# 10. Run tests
echo "Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -q --tb=no || echo "Some tests failed, check output above"
else
    echo "pytest not found, install with: pip install pytest"
fi

# 11. Check Git status
echo "Checking Git status..."
git status --porcelain

echo ""
echo "✅ Auto-fix script completed!"
echo ""
echo "📝 Next steps:"
echo "1. Review the changes: git diff"
echo "2. Commit the fixes: git add . && git commit -m 'fix: apply auto-fixes for publication readiness'"
echo "3. Create the release tag: git tag -a v0.4.0-md-validation -m 'Release v0.4.0 for manuscript validation'"
echo "4. Push changes: git push origin main --tags"
echo ""
echo "⚠️  Manual actions still required:"
echo "- Review and approve all changes"
echo "- Test the pipeline end-to-end"
echo "- Update README.md with verified commands"
echo "- Create comprehensive release notes" 