"""
AiForge Dashboard Backend API
Provides REST endpoints for the frontend to access LIMS data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import aiforge
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.aiforge.lims.experiment import ExperimentTracker
    from src.aiforge.lims.storage import StorageManager
    AIFORGE_AVAILABLE = True
except ImportError:
    AIFORGE_AVAILABLE = False
    print("Warning: AiForge LIMS not available. Using mock data.")

app = FastAPI(title="AiForge Dashboard API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class ExperimentStatus(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration: Optional[float] = None
    tags: List[str]
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress: Optional[float] = None
    error_message: Optional[str] = None

class SystemStatus(BaseModel):
    healthy: bool
    python_version: str
    workspace_path: str
    total_experiments: int
    running_experiments: int

# Mock data for when LIMS is not available
MOCK_EXPERIMENTS = [
    {
        "id": "EXP110801234",
        "name": "Qwen2.5-3B Code Fine-tune",
        "status": "running",
        "created_at": "2025-11-08T09:45:00Z",
        "started_at": "2025-11-08T09:46:00Z",
        "tags": ["code", "qwen", "production"],
        "current_epoch": 3,
        "total_epochs": 5,
        "progress": 60.0,
    },
    {
        "id": "EXP110725001",
        "name": "Llama-3.2-1B Math Reasoning",
        "status": "completed",
        "created_at": "2025-11-07T19:41:00Z",
        "started_at": "2025-11-07T19:41:00Z",
        "completed_at": "2025-11-07T22:15:00Z",
        "duration": 9240.0,
        "tags": ["math", "reasoning", "production"],
    },
]

@app.get("/")
async def root():
    return {
        "name": "AiForge Dashboard API",
        "version": "1.0.0",
        "aiforge_available": AIFORGE_AVAILABLE
    }

@app.get("/api/experiments", response_model=List[ExperimentStatus])
async def list_experiments(
    status: Optional[str] = None,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0
):
    """List all experiments with optional filtering"""

    if AIFORGE_AVAILABLE:
        try:
            # Use actual LIMS data
            tracker = ExperimentTracker()
            experiments = tracker.list_experiments()

            # Convert to API format
            result = []
            for exp in experiments:
                exp_data = {
                    "id": exp.experiment_id,
                    "name": exp.metadata.get("name", "Untitled Experiment"),
                    "status": exp.status,
                    "created_at": exp.created_at.isoformat() if hasattr(exp.created_at, 'isoformat') else exp.created_at,
                    "tags": exp.tags or [],
                }

                if exp.started_at:
                    exp_data["started_at"] = exp.started_at.isoformat() if hasattr(exp.started_at, 'isoformat') else exp.started_at

                if exp.completed_at:
                    exp_data["completed_at"] = exp.completed_at.isoformat() if hasattr(exp.completed_at, 'isoformat') else exp.completed_at

                if exp.duration:
                    exp_data["duration"] = exp.duration

                result.append(exp_data)

            # Apply status filter
            if status:
                result = [e for e in result if e["status"] == status]

            # Apply pagination
            return result[offset:offset + limit]

        except Exception as e:
            print(f"Error loading experiments from LIMS: {e}")
            # Fall back to mock data
            pass

    # Return mock data
    result = MOCK_EXPERIMENTS
    if status:
        result = [e for e in result if e["status"] == status]
    return result[offset:offset + limit]

@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get detailed information about a specific experiment"""

    if AIFORGE_AVAILABLE:
        try:
            tracker = ExperimentTracker()
            exp = tracker.get_experiment(experiment_id)

            if exp:
                return {
                    "id": exp.experiment_id,
                    "name": exp.metadata.get("name", "Untitled Experiment"),
                    "status": exp.status,
                    "created_at": exp.created_at.isoformat() if hasattr(exp.created_at, 'isoformat') else exp.created_at,
                    "tags": exp.tags or [],
                    "config": exp.config,
                    "metrics": exp.metrics,
                }
        except Exception as e:
            print(f"Error loading experiment {experiment_id}: {e}")

    # Mock data fallback
    for exp in MOCK_EXPERIMENTS:
        if exp["id"] == experiment_id:
            return exp

    raise HTTPException(status_code=404, detail="Experiment not found")

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status information"""

    workspace_path = os.environ.get("AIFORGE_WORKSPACE", "/Volumes/T7/AiForge")

    if AIFORGE_AVAILABLE:
        try:
            tracker = ExperimentTracker()
            experiments = tracker.list_experiments()
            running = [e for e in experiments if e.status == "running"]

            return {
                "healthy": True,
                "python_version": sys.version.split()[0],
                "workspace_path": workspace_path,
                "total_experiments": len(experiments),
                "running_experiments": len(running),
            }
        except Exception as e:
            print(f"Error getting system status: {e}")

    return {
        "healthy": True,
        "python_version": sys.version.split()[0],
        "workspace_path": workspace_path,
        "total_experiments": len(MOCK_EXPERIMENTS),
        "running_experiments": sum(1 for e in MOCK_EXPERIMENTS if e["status"] == "running"),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
