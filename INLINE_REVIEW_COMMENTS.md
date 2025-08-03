# Graph DiT-UQ Inline Review Comments

## Critical Issues (Must Fix)

### paper/main.tex
```latex
\begin{abstract}
TODO: Add abstract  # ❌ CRITICAL - Cannot submit without abstract
\end{abstract}

\section{Introduction}
TODO: Add introduction  # ❌ CRITICAL - Cannot submit without introduction

\section{Methods}
TODO: Add methods  # ❌ CRITICAL - Cannot submit without methods

\section{Results}
TODO: Add results  # ❌ CRITICAL - Cannot submit without results

\section{Discussion}
TODO: Add discussion  # ❌ CRITICAL - Cannot submit without discussion

\section{Conclusion}
TODO: Add conclusion  # ❌ CRITICAL - Cannot submit without conclusion
```
**Comment**: All sections contain TODO placeholders. This manuscript cannot be submitted to any journal. Need complete content for all sections.

### .github/workflows/ci.yml
```yaml
uses: actions/checkout@v3  # ❌ HIGH - Should pin to SHA256 digest
uses: actions/setup-python@v4  # ❌ HIGH - Should pin to SHA256 digest
```
**Comment**: Using version tags instead of SHA256 digests creates security risks. Pin to specific digests for reproducible and secure builds.

## Code Quality Issues

### src/rl/samplers.py:113
```python
replay_buffer = []  # ❌ MEDIUM - Missing type annotation
```
**Comment**: Add type annotation: `replay_buffer: list[Any] = []` or appropriate type.

### src/docking/quickvina2.py:73-74
```python
mol = Chem.EmbedMolecule(mol)  # ❌ MEDIUM - Module has no attribute
mol = Chem.MMFFOptimizeMolecule(mol)  # ❌ MEDIUM - Module has no attribute
```
**Comment**: These RDKit functions don't exist. Use `AllChem.EmbedMolecule()` and `AllChem.MMFFOptimizeMolecule()` instead.

### src/rl/rewards/sucb_pd.py:102
```python
pareto_front = np.array(pareto_front)  # ❌ MEDIUM - Type mismatch
```
**Comment**: Variable `pareto_front` is typed as `list[float] | None` but assigned `ndarray`. Fix type annotation.

### src/rl/rewards/ahi.py:171
```python
uncertainty = torch.tensor(uncertainty)  # ❌ MEDIUM - Type mismatch
```
**Comment**: Variable `uncertainty` is typed as `float` but assigned `Tensor`. Fix type annotation.

### src/rl/rewards/ahi.py:182
```python
reward = torch.tanh(uncertainty)  # ❌ MEDIUM - Argument type error
```
**Comment**: `uncertainty` is `float` but `torch.tanh` expects `Tensor`. Convert to tensor first.

## Test Issues

### tests/test_baselines_smoke.py
```python
# Multiple warnings about tensor size mismatches
# Target size (torch.Size([32, 1])) vs input size (torch.Size([32, 64/128/256]))
```
**Comment**: Fix tensor dimension mismatches in test data. Ensure target and input sizes match for each model.

### src/rl/rewards/ahi.py:191
```python
warnings.warn(f"AHI reward computation failed: {e}")  # ❌ MEDIUM - Test warning
```
**Comment**: AHI reward computation is failing in tests. Fix the sqrt() argument type issue.

## Documentation Issues

### README.md
```markdown
# Missing troubleshooting section
# Missing detailed installation guide
# Missing architecture documentation
```
**Comment**: Add comprehensive troubleshooting, detailed installation steps, and architecture overview for better usability.

## Security Issues

### Missing Security Scans
```bash
# No Trivy scan on Docker images
# No pip-audit on dependencies
# No detect-secrets scan
```
**Comment**: Run security scans before submission:
- `trivy image molecule-ai-base:latest`
- `pip-audit`
- `detect-secrets scan`

## Reproducibility Issues

### Missing End-to-End Test
```bash
# No complete pipeline test on clean environment
# No hash verification of stage4_results.parquet
# No WandB vs local metrics comparison
```
**Comment**: Test complete pipeline from Stage 1 to Stage 4 on clean VM to ensure reproducibility.

## Data Consistency Issues

### Manuscript vs Data
```latex
# No verification that LaTeX numbers match CSV outputs
# No cross-check of performance metrics
```
**Comment**: Cross-check all numeric values in manuscript with latest CSV data to ensure accuracy.

## Performance Issues

### Missing Benchmarks
```python
# No performance testing
# No resource monitoring
# No memory/GPU usage profiling
```
**Comment**: Add performance benchmarks and resource monitoring for production readiness.

## Git Hygiene Issues

### Untracked Files
```bash
?? PROJECT_SUMMARY.md  # ❌ LOW - Untracked file
```
**Comment**: Commit or ignore PROJECT_SUMMARY.md to clean Git status.

## Recommended Fixes

### Automated Fixes
```bash
# Fix ruff violations
source venv/bin/activate && ruff check --fix .

# Fix Black formatting (if needed)
source venv/bin/activate && black -l 88 .

# Clean Git status
git add PROJECT_SUMMARY.md && git commit -m "docs: add comprehensive project summary"
```

### Manual Fixes
1. **Complete LaTeX manuscript** - Replace all TODOs with actual content
2. **Fix type annotations** - Add proper type hints throughout codebase
3. **Fix tensor size mismatches** - Ensure consistent dimensions in tests
4. **Pin CI images** - Update GitHub Actions to use SHA256 digests
5. **Run security scans** - Execute Trivy, pip-audit, and detect-secrets
6. **Test end-to-end pipeline** - Validate on clean environment
7. **Cross-check data** - Verify manuscript numbers match CSV outputs
8. **Add documentation** - Create troubleshooting and architecture guides

## Priority Order
1. **CRITICAL**: Complete LaTeX manuscript
2. **CRITICAL**: Fix code quality issues
3. **HIGH**: Pin CI images to digests
4. **HIGH**: Run end-to-end pipeline test
5. **MEDIUM**: Address test warnings
6. **MEDIUM**: Run security scans
7. **LOW**: Add comprehensive documentation 