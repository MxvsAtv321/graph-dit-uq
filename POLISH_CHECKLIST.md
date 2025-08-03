# Graph DiT-UQ Final Polish Checklist for Journal Submission

## Repository Hygiene

- [x] **Tag Alignment**: Tag `v0.4.0-md-validation` exists and is properly aligned
- [x] **Large Files**: No files >10MB in Git history (largest: 5.4MB results file)
- [x] **Git LFS**: Large files properly handled (checkpoints moved to S3)
- [ ] **Git Status**: Clean working directory (1 untracked file: PROJECT_SUMMARY.md)
- [x] **Gitignore**: Comprehensive patterns for build artifacts, logs, checkpoints
- [x] **Dockerignore**: Proper exclusion of development files and large artifacts

## Code Quality & CI

- [x] **Black Formatting**: 100% compliance (69 files unchanged)
- [ ] **Ruff Violations**: 35 remaining violations (down from 279)
- [ ] **MyPy Type Checking**: 39 type errors in 10 files
- [x] **Test Coverage**: 35/36 tests passing (97% pass rate)
- [ ] **Test Warnings**: 10 warnings (tensor size mismatches, AHI reward issues)
- [ ] **CI Image Pinning**: GitHub Actions using @v3, @v4 tags (should use digests)
- [ ] **Secret Scanning**: No detect-secrets scan performed

## Reproducibility

- [x] **Requirements Lock**: requirements-lock.txt exists with exact versions
- [ ] **Environment Validation**: No clean VM test performed
- [ ] **Docker Stack Test**: No end-to-end pipeline validation
- [ ] **Hash Verification**: No stage4_results.parquet hash verification
- [ ] **WandB Metrics**: No local vs WandB comparison
- [ ] **Dependency Drift**: No requirements.txt vs pip freeze comparison

## Security & Compliance

- [ ] **Container Scanning**: No Trivy scan on Docker images
- [ ] **Dependency Audit**: No pip-audit scan for vulnerabilities
- [ ] **License Compliance**: No verification of external model licenses
- [x] **Security Policy**: SECURITY.md exists with disclosure policy
- [ ] **GPL Compliance**: No GPL v3 dependency check

## Figures & Data

- [x] **Figure Files**: PNG files exist in figures/ directory
- [ ] **Figure Rebuild**: No make figs command executed
- [ ] **Hash Verification**: No SHA-256 comparison of figures
- [ ] **SVG Optimization**: No SVG size/format validation
- [ ] **LaTeX Sync**: No figure filename verification with LaTeX

## Manuscript Consistency

- [ ] **LaTeX Structure**: Basic structure exists but incomplete (all TODOs)
- [ ] **Numeric Values**: No cross-check of LaTeX tables vs CSV data
- [ ] **Spell Check**: No LaTeX spell checking performed
- [ ] **PDF Compilation**: No latexmk -pdf test
- [ ] **Code Availability**: No update with tag and Docker digests
- [ ] **Abstract**: TODO placeholder
- [ ] **Introduction**: TODO placeholder
- [ ] **Methods**: TODO placeholder
- [ ] **Results**: TODO placeholder
- [ ] **Discussion**: TODO placeholder
- [ ] **Conclusion**: TODO placeholder

## Documentation

- [x] **README**: Comprehensive with quick start guide
- [x] **API Documentation**: Basic docstrings in place
- [ ] **Architecture Docs**: No detailed architecture documentation
- [ ] **Troubleshooting**: No common issues guide
- [ ] **Installation Guide**: Basic but could be more detailed

## Performance & Monitoring

- [x] **Pipeline Metrics**: WandB integration for experiment tracking
- [ ] **Resource Monitoring**: No memory/GPU usage profiling
- [ ] **Error Handling**: Basic error handling in place
- [ ] **Logging**: Comprehensive logging implemented
- [ ] **Performance Benchmarks**: No performance testing

## Final Validation

- [ ] **End-to-End Test**: No complete pipeline test on clean environment
- [ ] **Cross-Platform**: No testing on different OS
- [ ] **Dependency Conflicts**: No conflict resolution verification
- [ ] **Backward Compatibility**: No compatibility testing
- [ ] **Release Notes**: No comprehensive release notes

## Critical Issues (Must Fix Before Submission)

- [ ] **CRITICAL**: Complete LaTeX manuscript (all TODOs)
- [ ] **CRITICAL**: Fix 35 ruff violations
- [ ] **CRITICAL**: Resolve 39 mypy type errors
- [ ] **HIGH**: Pin CI images to SHA256 digests
- [ ] **HIGH**: Run end-to-end pipeline test
- [ ] **HIGH**: Cross-check numeric values in manuscript
- [ ] **MEDIUM**: Address test warnings
- [ ] **MEDIUM**: Run security scans
- [ ] **MEDIUM**: Verify figure reproducibility
- [ ] **LOW**: Add comprehensive documentation

## Auto-Fix Commands

```bash
# Fix ruff violations
source venv/bin/activate && ruff check --fix .

# Fix Black formatting (if needed)
source venv/bin/activate && black -l 88 .

# Create requirements lock
source venv/bin/activate && pip freeze > requirements-lock.txt

# Run tests
source venv/bin/activate && pytest tests/ -v

# Check for large files
find . -type f -size +10M -not -path "./.git/*" -not -path "./venv/*"

# Clean Git status
git add PROJECT_SUMMARY.md && git commit -m "docs: add comprehensive project summary"
```

## Manual Actions Required

1. **Complete LaTeX manuscript** - Replace all TODOs with actual content
2. **Run end-to-end pipeline test** - Validate on clean environment
3. **Security scanning** - Run Trivy and pip-audit
4. **Figure verification** - Rebuild and verify figure hashes
5. **Cross-check data** - Verify manuscript numbers match CSV outputs
6. **CI/CD hardening** - Pin images to digests
7. **Documentation polish** - Add troubleshooting and architecture docs

## Success Criteria

- [ ] 0 ruff violations
- [ ] 0 mypy errors
- [ ] 100% test pass rate
- [ ] Complete LaTeX manuscript
- [ ] Clean Git status
- [ ] All security scans passed
- [ ] End-to-end pipeline validated
- [ ] Figures reproducible
- [ ] Manuscript numbers verified
- [ ] CI/CD hardened 