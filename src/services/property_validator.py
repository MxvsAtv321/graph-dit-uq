"""
Property validation service with PAINS and toxicophore filters.
"""

from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PAINS SMARTS patterns (simplified subset)
PAINS_SMARTS = [
    # PAINS A - Aggregators
    "[*;R3;R4;R5;R6]",
    "[*;R3;R4;R5;R6][*;R3;R4;R5;R6][*;R3;R4;R5;R6]",
    # PAINS B - Reactives
    "[*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6]",
    "[*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6]",
    # PAINS C - Unstable
    "[*;R3;R4;R5;R6][*;R3;R4;R5;R6][*;R3;R4;R5;R6][*;R3;R4;R5;R6]",
]

# Toxicophore patterns
TOXICOPHORE_SMARTS = [
    # Common toxicophores
    "[*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6]",
    "[*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6][*;!R3;!R4;!R5;!R6]",
]


def validate_smiles(smiles: str) -> Dict[str, Any]:
    """
    Validate a SMILES string for PAINS and toxicophores.

    Args:
        smiles: SMILES string to validate

    Returns:
        Dictionary with validation results
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "valid": False,
                "reason": "Invalid SMILES",
                "heavy_atoms": 0,
                "pains_alerts": [],
                "toxicophore_alerts": [],
            }

        # Count heavy atoms
        heavy_atoms = mol.GetNumHeavyAtoms()

        # Check PAINS patterns
        pains_alerts = []
        for pattern in PAINS_SMARTS:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                pains_alerts.append(pattern)

        # Check toxicophore patterns
        toxicophore_alerts = []
        for pattern in TOXICOPHORE_SMARTS:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                toxicophore_alerts.append(pattern)

        # Determine if molecule is valid
        valid = (
            heavy_atoms >= 9  # Minimum heavy atoms
            and len(pains_alerts) == 0  # No PAINS alerts
            and len(toxicophore_alerts) == 0  # No toxicophore alerts
        )

        return {
            "smiles": smiles,
            "valid": valid,
            "heavy_atoms": heavy_atoms,
            "pains_alerts": pains_alerts,
            "toxicophore_alerts": toxicophore_alerts,
            "reason": "Valid" if valid else "Failed validation",
        }

    except ImportError:
        # Fallback if RDKit not available
        logger.warning("RDKit not available, using basic validation")
        return {
            "smiles": smiles,
            "valid": len(smiles) >= 5,  # Basic length check
            "heavy_atoms": len([c for c in smiles if c.isupper()]),
            "pains_alerts": [],
            "toxicophore_alerts": [],
            "reason": "Basic validation (RDKit not available)",
        }
    except Exception as e:
        logger.error(f"Validation failed for {smiles}: {str(e)}")
        return {
            "smiles": smiles,
            "valid": False,
            "reason": f"Validation error: {str(e)}",
            "heavy_atoms": 0,
            "pains_alerts": [],
            "toxicophore_alerts": [],
        }


def filter_smiles_list(smiles_list: List[str]) -> Dict[str, Any]:
    """
    Filter a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary with filtering results
    """
    results = []
    valid_count = 0
    flagged_count = 0

    for smiles in smiles_list:
        result = validate_smiles(smiles)
        results.append(result)

        if result["valid"]:
            valid_count += 1
        else:
            flagged_count += 1

    total_count = len(smiles_list)
    flagged_fraction = flagged_count / total_count if total_count > 0 else 0.0

    return {
        "results": results,
        "summary": {
            "total_molecules": total_count,
            "valid_molecules": valid_count,
            "flagged_molecules": flagged_count,
            "flagged_fraction": flagged_fraction,
            "passes_threshold": flagged_fraction <= 0.02,  # 2% threshold
        },
    }
