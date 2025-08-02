"""
QuickVina2 GPU service for molecular docking.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import time
import logging
import subprocess
import tempfile
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuickVina2 Service",
    description="GPU-accelerated molecular docking",
    version="1.0.0"
)

class DockingRequest(BaseModel):
    smiles: List[str]
    receptor_pdbqt: str = "/data/receptors/DDR1_receptor.pdbqt"
    exhaustiveness: int = 8
    num_modes: int = 1

class DockingResponse(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    gpu_available: bool

def check_gpu():
    """Check if GPU is available for docking."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except:
        return False

def run_quickvina2(smiles: str, receptor_path: str, exhaustiveness: int = 8) -> Dict[str, Any]:
    """
    Run QuickVina2 docking for a single molecule.
    
    Args:
        smiles: SMILES string of the molecule
        receptor_path: Path to receptor PDBQT file
        exhaustiveness: Docking exhaustiveness parameter
        
    Returns:
        Dictionary with docking results
    """
    try:
        # Mock QuickVina2 implementation
        # In production, this would:
        # 1. Convert SMILES to 3D structure (e.g., using RDKit)
        # 2. Generate PDBQT file
        # 3. Run QuickVina2 command
        # 4. Parse output for binding affinity
        
        # Simulate docking time based on molecule complexity
        mol_length = len(smiles)
        docking_time = min(2.0 + mol_length * 0.1, 10.0)  # 2-10 seconds
        
        # Generate realistic docking score based on SMILES
        np.random.seed(hash(smiles) % 2**32)
        
        # Base score logic
        if "C1=CC=CC=C1" in smiles:  # Benzene-like
            base_score = -6.5
        elif "CCO" in smiles:  # Alcohol-like
            base_score = -5.8
        elif "CC(C)C" in smiles:  # Branched
            base_score = -7.2
        else:
            base_score = -6.0 + np.random.normal(0, 1.5)
        
        # Add some variation
        final_score = base_score + np.random.normal(0, 0.5)
        
        return {
            "smiles": smiles,
            "binding_affinity": float(final_score),
            "rmsd_lb": float(np.random.uniform(0.5, 2.0)),
            "rmsd_ub": float(np.random.uniform(2.0, 4.0)),
            "docking_time_s": docking_time,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Docking failed for {smiles}: {str(e)}")
        return {
            "smiles": smiles,
            "binding_affinity": None,
            "rmsd_lb": None,
            "rmsd_ub": None,
            "docking_time_s": 0.0,
            "status": "failed",
            "error": str(e)
        }

@app.get("/live", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers."""
    gpu_available = check_gpu()
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        gpu_available=gpu_available
    )

@app.post("/dock", response_model=DockingResponse)
async def dock_molecules(request: DockingRequest):
    """
    Dock molecules against a receptor.
    
    Args:
        request: DockingRequest with SMILES and parameters
        
    Returns:
        DockingResponse with binding affinities and metadata
    """
    try:
        start_time = time.time()
        logger.info(f"Docking {len(request.smiles)} molecules")
        
        # Check if receptor file exists
        if not os.path.exists(request.receptor_pdbqt):
            raise HTTPException(
                status_code=400, 
                detail=f"Receptor file not found: {request.receptor_pdbqt}"
            )
        
        # Process molecules in batches for better performance
        batch_size = 10
        all_results = []
        total_docking_time = 0.0
        
        for i in range(0, len(request.smiles), batch_size):
            batch = request.smiles[i:i+batch_size]
            batch_start = time.time()
            
            # Process batch
            batch_results = []
            for smiles in batch:
                result = run_quickvina2(
                    smiles, 
                    request.receptor_pdbqt, 
                    request.exhaustiveness
                )
                batch_results.append(result)
                total_docking_time += result.get("docking_time_s", 0.0)
            
            all_results.extend(batch_results)
            batch_time = time.time() - batch_start
            logger.info(f"Processed batch {i//batch_size + 1} in {batch_time:.2f}s")
        
        # Calculate success rate
        successful_docks = sum(1 for r in all_results if r["status"] == "success")
        success_rate = successful_docks / len(all_results) if all_results else 0.0
        
        # Fail if success rate is too low
        if success_rate < 0.8:
            raise HTTPException(
                status_code=500,
                detail=f"Docking success rate too low: {success_rate:.1%} < 80%"
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Docked {len(all_results)} molecules in {processing_time:.2f}s")
        
        return DockingResponse(
            results=all_results,
            metadata={
                "n_molecules": len(all_results),
                "success_rate": success_rate,
                "processing_time_s": processing_time,
                "total_docking_time_s": total_docking_time,
                "receptor": request.receptor_pdbqt,
                "exhaustiveness": request.exhaustiveness,
                "gpu_available": check_gpu()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Docking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Docking failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "QuickVina2",
        "version": "1.0.0",
        "gpu_available": check_gpu(),
        "endpoints": {
            "health": "/live",
            "dock": "/dock"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5678) 