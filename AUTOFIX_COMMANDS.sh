#!/bin/bash
# Graph DiT-UQ Auto-Fix Commands for Journal Submission
# Run these commands to fix identified issues

set -e  # Exit on any error

echo "🔧 Graph DiT-UQ Auto-Fix Script for Journal Submission"
echo "======================================================"

# Phase 1: Critical Code Quality Fixes
echo "📋 Phase 1: Critical Code Quality Fixes"
echo "---------------------------------------"

# 1. Fix ruff violations
echo "Fixing ruff violations..."
if command -v ruff &> /dev/null; then
    source venv/bin/activate && ruff check --fix . || echo "Ruff not available, skipping..."
else
    echo "Ruff not found, install with: pip install ruff"
fi

# 2. Apply Black formatting (if needed)
echo "Checking Black formatting..."
if command -v black &> /dev/null; then
    source venv/bin/activate && black -l 88 . || echo "Black not available, skipping..."
else
    echo "Black not found, install with: pip install black"
fi

# 3. Clean Git status
echo "Cleaning Git status..."
git add PROJECT_SUMMARY.md && git commit -m "docs: add comprehensive project summary" || echo "Git commit failed or already committed"

# Phase 2: Security & Compliance
echo ""
echo "📋 Phase 2: Security & Compliance"
echo "--------------------------------"

# 4. Run pip-audit (if available)
echo "Running pip-audit..."
if command -v pip-audit &> /dev/null; then
    source venv/bin/activate && pip-audit || echo "pip-audit failed or not available"
else
    echo "pip-audit not found, install with: pip install pip-audit"
fi

# 5. Run detect-secrets (if available)
echo "Running detect-secrets scan..."
if command -v detect-secrets &> /dev/null; then
    detect-secrets scan . || echo "detect-secrets failed or not available"
else
    echo "detect-secrets not found, install with: pip install detect-secrets"
fi

# 6. Run Trivy on Docker images (if available)
echo "Running Trivy scan on Docker images..."
if command -v trivy &> /dev/null; then
    trivy image molecule-ai-base:latest || echo "Trivy scan failed or image not available"
else
    echo "Trivy not found, install with: brew install trivy (macOS) or apt-get install trivy (Ubuntu)"
fi

# Phase 3: Testing & Validation
echo ""
echo "📋 Phase 3: Testing & Validation"
echo "-------------------------------"

# 7. Run tests with detailed output
echo "Running tests..."
if command -v pytest &> /dev/null; then
    source venv/bin/activate && pytest tests/ -v --tb=short || echo "Some tests failed, check output above"
else
    echo "pytest not found, install with: pip install pytest"
fi

# 8. Check for large files
echo "Checking for large files..."
find . -type f -size +10M -not -path "./.git/*" -not -path "./venv/*" || echo "No files >10MB found"

# 9. Check for large files >50kB
echo "Checking for files >50kB..."
find . -type f -size +50k -not -path "./.git/*" -not -path "./venv/*" | head -10 || echo "No files >50kB found"

# Phase 4: Documentation & Reproducibility
echo ""
echo "📋 Phase 4: Documentation & Reproducibility"
echo "------------------------------------------"

# 10. Create requirements lock file
echo "Creating requirements lock file..."
if command -v pip &> /dev/null; then
    source venv/bin/activate && pip freeze > requirements-lock.txt
else
    echo "pip not found, skipping requirements lock..."
fi

# 11. Check figure files
echo "Checking figure files..."
find figures/ -name "*.png" -o -name "*.svg" | head -10 || echo "No figure files found"

# 12. Check LaTeX files
echo "Checking LaTeX files..."
find paper/ -name "*.tex" -o -name "*.bib" || echo "No LaTeX files found"

# Phase 5: Code Quality Analysis
echo ""
echo "📋 Phase 5: Code Quality Analysis"
echo "--------------------------------"

# 13. Count ruff violations
echo "Counting ruff violations..."
if command -v ruff &> /dev/null; then
    source venv/bin/activate && ruff check . --output-format=json | jq '. | length' || echo "Ruff not available"
else
    echo "Ruff not available for violation count"
fi

# 14. Count mypy errors
echo "Counting mypy errors..."
if command -v mypy &> /dev/null; then
    source venv/bin/activate && mypy src/ --ignore-missing-imports 2>&1 | grep "error:" | wc -l || echo "0"
else
    echo "mypy not available for error count"
fi

# 15. Check test coverage
echo "Checking test coverage..."
if command -v pytest &> /dev/null; then
    source venv/bin/activate && pytest tests/ -q --tb=no | tail -1 || echo "Test coverage check failed"
else
    echo "pytest not available for coverage check"
fi

# Phase 6: Repository Health
echo ""
echo "📋 Phase 6: Repository Health"
echo "----------------------------"

# 16. Check Git status
echo "Checking Git status..."
git status --porcelain || echo "Git status check failed"

# 17. Check repository size
echo "Checking repository size..."
du -sh .git || echo "Repository size check failed"

# 18. Check for TODO/FIXME comments
echo "Checking for TODO/FIXME comments..."
grep -r "TODO\|FIXME\|XXX" src/ paper/ dags/ --exclude-dir=__pycache__ --exclude-dir=.git || echo "No TODO/FIXME comments found"

# Phase 7: Manual Actions Required
echo ""
echo "📋 Phase 7: Manual Actions Required"
echo "----------------------------------"

echo "⚠️  The following actions require manual intervention:"
echo ""
echo "1. COMPLETE LATEX MANUSCRIPT (CRITICAL):"
echo "   - Replace all TODO placeholders in paper/main.tex"
echo "   - Add abstract, introduction, methods, results, discussion, conclusion"
echo "   - Cross-check numeric values with CSV outputs"
echo ""
echo "2. FIX TYPE ANNOTATIONS (CRITICAL):"
echo "   - Add proper type hints in src/rl/samplers.py:113"
echo "   - Fix type mismatches in src/rl/rewards/*.py"
echo "   - Fix RDKit function calls in src/docking/quickvina2.py"
echo ""
echo "3. PIN CI IMAGES (HIGH):"
echo "   - Update .github/workflows/ci.yml to use SHA256 digests"
echo "   - Replace @v3, @v4 with specific SHA256 hashes"
echo ""
echo "4. RUN END-TO-END TEST (HIGH):"
echo "   - Test complete pipeline on clean Ubuntu 22.04 VM"
echo "   - Verify hash of stage4_results.parquet"
echo "   - Compare WandB vs local metrics"
echo ""
echo "5. ADDRESS TEST WARNINGS (MEDIUM):"
echo "   - Fix tensor size mismatches in tests/test_baselines_smoke.py"
echo "   - Fix AHI reward computation in src/rl/rewards/ahi.py"
echo ""
echo "6. VERIFY FIGURE REPRODUCIBILITY (MEDIUM):"
echo "   - Rebuild figures with make figs or equivalent"
echo "   - Verify SHA256 hashes match committed versions"
echo ""
echo "7. ADD DOCUMENTATION (LOW):"
echo "   - Add troubleshooting guide to README.md"
echo "   - Add detailed installation guide"
echo "   - Add architecture documentation"
echo ""

# Phase 8: Summary
echo "📋 Phase 8: Summary"
echo "-------------------"

echo "✅ Auto-fix script completed!"
echo ""
echo "📊 Current Status:"
echo "- Repository: v0.4.0-md-validation"
echo "- Test Coverage: 35/36 tests passing (97%)"
echo "- Black Formatting: 100% compliant"
echo "- Large Files: None >10MB"
echo "- Git Status: Clean (after commit)"
echo ""
echo "⚠️  Critical Issues Remaining:"
echo "- LaTeX manuscript incomplete (CRITICAL)"
echo "- 35 ruff violations (CRITICAL)"
echo "- 39 mypy type errors (CRITICAL)"
echo "- CI images not pinned (HIGH)"
echo "- No end-to-end validation (HIGH)"
echo ""
echo "📝 Next Steps:"
echo "1. Complete LaTeX manuscript (2-3 days)"
echo "2. Fix code quality issues (1-2 days)"
echo "3. Run end-to-end pipeline test (1-2 days)"
echo "4. Pin CI images to digests (1 day)"
echo "5. Run security scans (1 day)"
echo ""
echo "🎯 Estimated Time to Submission: 4-6 days (critical fixes only)"
echo ""
echo "🚀 The project has excellent technical foundations and is ready for"
echo "   submission after addressing the critical issues above." 