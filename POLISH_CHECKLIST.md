# Graph DiT-UQ Publication Readiness Checklist

## Repository & Git Hygiene

- [ ] **Tag Creation**: Create annotated tag `v0.4.0-md-validation` at HEAD of main
- [ ] **Large Files**: Move files >10MB to S3 (found: `checkpoints/graph_dit_10k.pt` 76MB, `logs/dag_processor_manager/dag_processor_manager.log` 6MB)
- [ ] **Large Data Files**: Move parquet/csv/png files >50kB to S3 (found: `screenshots/early_pareto.png` 664K, `screenshots/uncertainty_pareto.png` 792K)
- [ ] **Missing .dockerignore**: Create `.dockerignore` file to exclude build artifacts
- [ ] **Incomplete .gitignore**: Add missing patterns (`*.pt`, `.pytest_cache`, `*.DS_Store`, `logs/`, `venv/`, `__pycache__/`)
- [ ] **Log Files**: Remove or ignore Airflow log files from Git history (found: 6MB+ of log files)
- [ ] **Untracked Files**: Review and commit/ignore untracked files in `scripts/` and `logs/`
- [ ] **Git Status**: Clean working directory before tagging

## Code Quality & CI

- [ ] **Code Formatting**: Apply Black formatting to all Python files (found: formatting inconsistencies)
- [ ] **Linting Issues**: Fix 279 ruff violations across codebase
- [ ] **Unused Imports**: Remove unused imports (found: torch imports in DAG files)
- [ ] **Type Hints**: Add mypy type hints to source files
- [ ] **CI Pinning**: Pin GitHub Actions images by digest instead of tags (currently using @v3, @v4)
- [ ] **Test Coverage**: Ensure >95% test pass rate (current: 35/36 tests passing ✅)
- [ ] **Docker Security**: Ensure Dockerfiles use non-root UID drop
- [ ] **Secrets Scan**: Run detect-secrets scan for hardcoded credentials

## Reproducibility & Pipeline Integrity

- [ ] **Environment Lock**: Create `requirements-lock.txt` for exact dependency versions
- [ ] **DAG Validation**: Test all DAGs (Stage 1-4) on clean environment
- [ ] **Data Consistency**: Verify WandB metrics match local CSV outputs
- [ ] **Figure Reproducibility**: Regenerate all figures and verify PNG hashes
- [ ] **Documentation Sync**: Update README.md with actual working commands
- [ ] **Docker Build**: Test Docker image builds on clean environment

## Security & Licensing

- [ ] **Container Scanning**: Run Trivy scan on Docker images for CVEs
- [ ] **Dependency Audit**: Run pip-audit for security vulnerabilities
- [ ] **License Compliance**: Verify 3rd-party model licenses (DiffDock-L, QuickVina)
- [ ] **SECURITY.md**: Create security disclosure policy
- [ ] **GPL Compliance**: Check for GPLv3 transitive dependencies

## Figures & Data Artifacts

- [ ] **Figure Regeneration**: Run `make figs` or equivalent to regenerate all figures
- [ ] **SVG Optimization**: Ensure SVGs are text-based and <1MB
- [ ] **Caption Sync**: Verify LaTeX figure captions match actual file names
- [ ] **Large Screenshots**: Move large screenshot files to S3
- [ ] **Figure Quality**: Ensure publication-quality resolution (300 DPI)

## Manuscript Sync

- [ ] **LaTeX Structure**: Create proper LaTeX manuscript structure in `paper/`
- [ ] **Number Consistency**: Cross-check LaTeX tables with latest CSV data
- [ ] **Spell Check**: Run language tool on LaTeX content
- [ ] **TODO Cleanup**: Remove all "TODO", "??", "FIXME" from manuscript
- [ ] **PDF Build**: Build PDF with `latexmk -pdf paper/main.tex` (zero warnings)
- [ ] **Code Availability**: Update to cite tag `v0.4.0-md-validation` and Docker digests

## Documentation & README

- [ ] **Quick Start**: Verify README.md quick start commands work
- [ ] **Installation**: Test fresh clone and setup on clean machine
- [ ] **API Documentation**: Add docstrings and type hints to all public APIs
- [ ] **Architecture**: Update architecture diagram and service descriptions
- [ ] **Troubleshooting**: Add common issues and solutions to README

## Performance & Monitoring

- [ ] **Memory Usage**: Profile memory usage of large data processing
- [ ] **GPU Utilization**: Monitor GPU usage in production runs
- [ ] **Pipeline Metrics**: Add performance monitoring to Airflow DAGs
- [ ] **Error Handling**: Improve error handling and logging throughout
- [ ] **Resource Limits**: Set appropriate resource limits for containers

## Final Validation

- [ ] **End-to-End Test**: Run complete pipeline from Stage 1 to Stage 4
- [ ] **Cross-Platform**: Test on different OS (Ubuntu, macOS)
- [ ] **Dependency Conflicts**: Resolve any dependency version conflicts
- [ ] **Backward Compatibility**: Ensure changes don't break existing functionality
- [ ] **Release Notes**: Create comprehensive release notes for v0.4.0

## Critical Issues (Must Fix Before Publication)

- [ ] **CRITICAL**: Remove 76MB checkpoint file from Git history
- [ ] **CRITICAL**: Create .dockerignore to prevent build artifact inclusion
- [ ] **HIGH**: Fix 279 ruff violations for code quality
- [ ] **HIGH**: Apply Black formatting to all Python files
- [ ] **HIGH**: Remove Airflow log files from Git tracking
- [ ] **MEDIUM**: Pin CI/CD images to specific digests
- [ ] **MEDIUM**: Create environment lock file for reproducibility 