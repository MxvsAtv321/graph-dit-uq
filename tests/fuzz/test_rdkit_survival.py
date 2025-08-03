"""
Red-team test cases for RDKit survival testing.
Tests pathological SMILES that could crash or hang cheminformatics tools.
"""

import pytest
import signal
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


class TimeoutError(Exception):
    """Custom timeout exception"""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation exceeded time limit")


# Pathological SMILES designed to stress-test cheminformatics tools
REDTEAM_SMILES = [
    # 1. Deeply nested ring systems - exponential symmetry detection
    "C1CCC2(CCC3(CCC4(CCC5(CCC6(CCC7(CCC8(CCC9(CCC%10(CCC1%10)C9)C8)C7)C6)C5)C4)C3)C2",
    # Expected: Symmetry perception timeout in RDKit < 2023.09
    # 2. Maximum branch depth - stack overflow in recursive SMARTS
    "C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C)))))))))))))))))))))Br",
    # Expected: RecursionError in substructure matching
    # 3. Alternating stereochemistry - exponential stereoisomers
    "C[C@H]1C[C@@H](C[C@H](C[C@@H](C[C@H](C[C@@H](C[C@H](C[C@@H](C1)C)C)C)C)C)C)C",
    # Expected: Memory exhaustion in stereochemistry enumeration
    # 4. Extreme ring sizes - breaks ring perception heuristics
    "C1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1",
    # Expected: Ring perception gives up, undefined behavior
    # 5. Pathological aromaticity - contradictory aromatic assignments
    "c1ccc2c(c1)c1ccccc1c1c2cccc1-c1ccccc1",
    # Expected: Kekulization failure, aromatic system inconsistent
    # 6. Maximum valence violation hidden in SMILES
    "C(C)(C)(C)(C)(C)(C)(C)(C)C",
    # Expected: Sanitization should fail but some parsers miss it
    # 7. Disconnected fragments with contradictory charges
    "[CH3+].[CH3-].[CH3+].[CH3-].[CH3+].[CH3-]",
    # Expected: Charge balancing algorithms fail
    # 8. Macrocycle with all possible stereoisomers
    "C[C@]12CC[C@@]3(C)[C@]4(C)CC[C@@]5(C)[C@]6(C)CC[C@@]7(C)[C@]8(C)CC[C@@]1(C)[C@]2(C)CC[C@@]3(C)[C@]4(C)CC[C@@]5(C)[C@]6(C)CC[C@@]78C",
    # Expected: Conformer generation hangs or produces 2^n conformers
    # 9. Invalid implicit hydrogen count
    "C12C3C4C1C5C2C6C3C7C4C8C5C9C6C%10C7C%11C8C%12C9C%10C%11C%12",
    # Expected: Implicit H calculation integer overflow
    # 10. Quadruple bond in supposedly valid syntax
    "C####C",
    # Expected: Parser accepts but molecule is chemically impossible
]


def check_smiles_parsing_safety(smiles, timeout=5):
    """
    Test if SMILES parsing is safe with timeout.

    Args:
        smiles: SMILES string to test
        timeout: Timeout in seconds

    Returns:
        Test result dictionary
    """
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    result = {
        "smiles": smiles,
        "parse_success": False,
        "sanitize_success": False,
        "operations_success": False,
        "timeout": False,
        "error_type": None,
        "error_message": None,
    }

    try:
        # Test 1: Basic parsing
        mol = Chem.MolFromSmiles(smiles)
        result["parse_success"] = mol is not None

        if mol:
            # Test 2: Sanitization
            try:
                Chem.SanitizeMol(mol)
                result["sanitize_success"] = True
            except Exception as e:
                result["error_type"] = "sanitization"
                result["error_message"] = str(e)

            # Test 3: Common operations
            if result["sanitize_success"]:
                try:
                    # Ring info
                    mol.GetRingInfo()

                    # Aromatic rings
                    rdMolDescriptors.CalcNumAromaticRings(mol)

                    # Conformer generation (limited)
                    AllChem.EmbedMolecule(mol, maxAttempts=1)

                    result["operations_success"] = True
                except Exception as e:
                    result["error_type"] = "operations"
                    result["error_message"] = str(e)

        signal.alarm(0)

    except TimeoutError:
        result["timeout"] = True
        result["error_type"] = "timeout"
        signal.alarm(0)
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        signal.alarm(0)

    return result


@pytest.mark.parametrize("smiles", REDTEAM_SMILES)
def test_rdkit_survival(smiles):
    """
    Test RDKit survival against pathological SMILES.

    Args:
        smiles: Pathological SMILES string
    """
    result = check_smiles_parsing_safety(smiles, timeout=10)

    # Assert that RDKit doesn't crash or hang
    assert not result["timeout"], f"RDKit timed out on SMILES: {smiles}"

    # Log the result for analysis
    print(f"SMILES: {smiles}")
    print(f"  Parse: {result['parse_success']}")
    print(f"  Sanitize: {result['sanitize_success']}")
    print(f"  Operations: {result['operations_success']}")
    if result["error_type"]:
        print(f"  Error: {result['error_type']} - {result['error_message']}")
    print()


def test_batch_processing_safety():
    """
    Test batch processing of pathological SMILES.
    """
    print("Testing batch processing safety...")

    results = []
    for i, smiles in enumerate(REDTEAM_SMILES):
        result = check_smiles_parsing_safety(smiles, timeout=5)
        result["index"] = i
        results.append(result)

    # Count failures
    timeouts = sum(1 for r in results if r["timeout"])
    parse_failures = sum(1 for r in results if not r["parse_success"])
    sanitize_failures = sum(1 for r in results if not r["sanitize_success"])

    print("Batch processing results:")
    print(f"  Total SMILES: {len(results)}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Parse failures: {parse_failures}")
    print(f"  Sanitize failures: {sanitize_failures}")

    # Assert reasonable failure rates
    assert timeouts <= 2, f"Too many timeouts: {timeouts}"
    assert parse_failures <= 5, f"Too many parse failures: {parse_failures}"


def test_memory_usage():
    """
    Test memory usage during pathological SMILES processing.
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Initial memory usage: {initial_memory:.1f} MB")

    # Process all pathological SMILES
    for smiles in REDTEAM_SMILES:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                Chem.SanitizeMol(mol)
        except Exception:
            pass  # Expected failures

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    print(f"Final memory usage: {final_memory:.1f} MB")
    print(f"Memory increase: {memory_increase:.1f} MB")

    # Assert reasonable memory usage
    assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f} MB"


if __name__ == "__main__":
    # Run tests manually
    print("Running RDKit survival tests...")

    for smiles in REDTEAM_SMILES:
        result = check_smiles_parsing_safety(smiles, timeout=5)
        print(f"SMILES: {smiles}")
        print(f"  Result: {result}")
        print()

    print("All tests completed.")
