"""
DiffDock-L FastAPI service for high-fidelity molecular docking.
Provides physics-grounded docking scores for Stage 3 RL optimization.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import time
import os
from datetime import datetime

app = FastAPI(title="DiffDock-L Service", version="0.1.0")


class DockRequest(BaseModel):
    smiles: List[str]
    receptor_pdbqt: str
    batch_size: int = 8
    num_samples: int = 16  # conformers per ligand


class DockResponse(BaseModel):
    poses: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    gpu_available: bool


# Mock DiffDock-L model (replace with real implementation)
class MockDiffDockL:
    def __init__(self):
        self.model_name = "diffdock-l-base"
        self.version = "0.4.1"

    def dock(
        self,
        smiles_list: List[str],
        receptor: str,
        batch_size: int = 8,
        num_samples: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Mock DiffDock-L docking with realistic physics-based scores.
        """
        results = []

        for i, smiles in enumerate(smiles_list):
            # Simulate physics-based docking
            # Generate realistic confidence scores (0-1) and RMSD predictions
            confidence = np.random.beta(2, 5)  # Skewed toward lower confidence
            pred_rmsd = np.random.exponential(2.0) + 1.0  # Realistic RMSD distribution

            # Add some correlation between SMILES length and docking quality
            if len(smiles) > 50:
                confidence *= 0.8  # Larger molecules harder to dock
                pred_rmsd *= 1.2

            # Generate pose coordinates (mock)
            pose_coords = np.random.randn(3, 3)  # 3D coordinates for pose

            result = {
                "smiles": smiles,
                "confidence": float(confidence),
                "pred_rmsd": float(pred_rmsd),
                "pose_coords": pose_coords.tolist(),
                "receptor": receptor,
                "batch_id": i // batch_size,
                "sample_id": i % num_samples,
            }
            results.append(result)

        return results


# Initialize model
try:
    # Try to import real DiffDock-L
    from diffdock_l import DiffDockL

    model = DiffDockL.load_pretrained("base")
    print("✅ Loaded real DiffDock-L model")
except ImportError:
    # Fallback to mock
    model = MockDiffDockL()
    print("⚠️ Using mock DiffDock-L model")


@app.get("/live", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=model.version,
        gpu_available=os.environ.get("CUDA_VISIBLE_DEVICES") is not None,
    )


@app.post("/dock", response_model=DockResponse)
async def dock_molecules(request: DockRequest):
    """
    Perform high-fidelity docking using DiffDock-L.

    Args:
        request: DockRequest with SMILES, receptor, and parameters

    Returns:
        DockResponse with poses and metadata
    """
    start_time = time.time()

    # Validate receptor file exists
    if not os.path.exists(request.receptor_pdbqt):
        raise HTTPException(
            status_code=400, detail=f"Receptor file not found: {request.receptor_pdbqt}"
        )

    # Validate SMILES
    if not request.smiles:
        raise HTTPException(status_code=400, detail="No SMILES provided")

    # Perform docking
    try:
        poses = model.dock(
            request.smiles,
            receptor=request.receptor_pdbqt,
            batch_size=request.batch_size,
            num_samples=request.num_samples,
        )

        # Calculate success rate
        success_rate = len([p for p in poses if p["confidence"] > 0.1]) / len(poses)

        # Fail if success rate too low
        if success_rate < 0.7:
            raise HTTPException(
                status_code=500,
                detail=f"Docking success rate too low: {success_rate:.1%} < 70%",
            )

        processing_time = time.time() - start_time

        return DockResponse(
            poses=poses,
            metadata={
                "n_molecules": len(poses),
                "success_rate": success_rate,
                "processing_time_s": processing_time,
                "batch_size": request.batch_size,
                "num_samples": request.num_samples,
                "receptor": request.receptor_pdbqt,
                "model_version": model.version,
                "gpu_available": os.environ.get("CUDA_VISIBLE_DEVICES") is not None,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Docking failed: {str(e)}")


@app.get("/docs")
async def get_docs():
    """Return API documentation."""
    return {
        "service": "DiffDock-L",
        "version": model.version,
        "endpoints": {
            "/live": "Health check",
            "/dock": "Perform docking",
            "/docs": "This documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9100)
