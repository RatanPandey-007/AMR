"""FastAPI Backend Server for AMR-Predict.

Exposes high-performance RESTful endpoints for multi-antibiotic resistance prediction,
local/global SHAP explainability, model comparison benchmarks, and dataset EDA.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    VALID_CATEGORIES,
    NUMERICAL_BOUNDS,
    FEATURE_DISPLAY_NAMES,
    RISK_THRESHOLDS,
    MEDICAL_DISCLAIMER,
    MODEL_ATTRIBUTION_NOTICE,
    ANTIBIOTIC_TARGET_MAP,
)
from src.predict import predict_patient, load_inference_artifacts
from src.explain import explain_patient_prediction, get_global_shap_summary
from src.preprocessing import DataValidationError

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AMR_API")

from contextlib import asynccontextmanager

# Pydantic Schemas
class PatientInputSchema(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Patient age in years (0-120)")
    sex: str = Field(..., description="Biological sex ('F' or 'M')")
    infection_type: str = Field(..., description="Infection syndrome category")
    organism: str = Field(..., description="Isolated microbial pathogen")
    diabetes: int = Field(0, ge=0, le=1, description="Diabetes mellitus (0 or 1)")
    recent_hospitalization_90d: int = Field(0, ge=0, le=1, description="Hospital admission in last 90d (0 or 1)")
    recent_antibiotic_use_90d: int = Field(0, ge=0, le=1, description="Antibiotic therapy in last 90d (0 or 1)")
    num_prior_uti_1yr: int = Field(0, ge=0, le=50, description="Documented UTIs in past 1 year")
    catheter_use: int = Field(0, ge=0, le=1, description="Presence of indwelling urinary catheter (0 or 1)")
    immunocompromised: int = Field(0, ge=0, le=1, description="Immunocompromised condition (0 or 1)")
    nursing_home_resident: int = Field(0, ge=0, le=1, description="Resident of long-term care facility (0 or 1)")
    prior_resistant_culture_1yr: int = Field(0, ge=0, le=1, description="History of resistant culture in past year (0 or 1)")
    creatinine_mg_dl: float = Field(..., ge=0.1, le=25.0, description="Serum creatinine in mg/dL")
    wbc_count_k_ul: float = Field(..., ge=0.1, le=100.0, description="White blood cell count (k/µL)")
    travel_last_6mo: int = Field(0, ge=0, le=1, description="International travel in last 6 months (0 or 1)")
    healthcare_worker: int = Field(0, ge=0, le=1, description="Healthcare occupational exposure (0 or 1)")


class ExplainRequestSchema(BaseModel):
    patient_data: PatientInputSchema
    antibiotic: str = Field(..., description="Target antibiotic to explain (e.g. 'Ciprofloxacin')")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Pre-loading machine learning models into memory...")
        load_inference_artifacts()
        logger.info("All 8 AMR models and preprocessor successfully loaded.")
    except Exception as e:
        logger.warning(f"Models not ready on startup: {e}")
    yield


app = FastAPI(
    title="AMR-Predict API",
    description="AI-Powered Antimicrobial Resistance Prediction & Explainability System (Research Prototype)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
STATIC_DIR = BASE_DIR / "app" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Endpoints
@app.get("/")
def serve_index():
    """Serves the main AMR-Predict web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "AMR-Predict API is online. UI static files not found."})


@app.get("/api/health")
def get_health():
    """System health check and loaded model status."""
    reg_path = ARTIFACTS_DIR / "model_registry.json"
    is_ready = reg_path.exists()
    return {
        "status": "healthy",
        "models_ready": is_ready,
        "timestamp": time.time(),
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@app.get("/api/config")
def get_system_config():
    """Returns clinical categories, ranges, display names, presets, and safety disclaimers."""
    presets = [
        {
            "id": "preset_elderly_catheter",
            "name": "Elderly Catheter-Associated UTI (High Risk)",
            "description": "78yo female, nursing home resident with catheter and recent antibiotic use.",
            "data": {
                "age": 78,
                "sex": "F",
                "infection_type": "Catheter_Associated_UTI",
                "organism": "Klebsiella_pneumoniae",
                "diabetes": 1,
                "recent_hospitalization_90d": 1,
                "recent_antibiotic_use_90d": 1,
                "num_prior_uti_1yr": 3,
                "catheter_use": 1,
                "immunocompromised": 0,
                "nursing_home_resident": 1,
                "prior_resistant_culture_1yr": 1,
                "creatinine_mg_dl": 1.65,
                "wbc_count_k_ul": 14.8,
                "travel_last_6mo": 0,
                "healthcare_worker": 0,
            }
        },
        {
            "id": "preset_young_uncomplicated",
            "name": "Young Uncomplicated E. coli UTI (Low Risk)",
            "description": "24yo female with first episode of acute uncomplicated cystitis and no prior exposures.",
            "data": {
                "age": 24,
                "sex": "F",
                "infection_type": "UTI",
                "organism": "E_coli",
                "diabetes": 0,
                "recent_hospitalization_90d": 0,
                "recent_antibiotic_use_90d": 0,
                "num_prior_uti_1yr": 0,
                "catheter_use": 0,
                "immunocompromised": 0,
                "nursing_home_resident": 0,
                "prior_resistant_culture_1yr": 0,
                "creatinine_mg_dl": 0.85,
                "wbc_count_k_ul": 7.2,
                "travel_last_6mo": 0,
                "healthcare_worker": 0,
            }
        },
        {
            "id": "preset_complicated_pseudomonas",
            "name": "Complicated Pseudomonas UTI in Diabetic Male",
            "description": "64yo male with diabetes, previous resistant culture, and elevated creatinine.",
            "data": {
                "age": 64,
                "sex": "M",
                "infection_type": "Complicated_UTI",
                "organism": "Pseudomonas_aeruginosa",
                "diabetes": 1,
                "recent_hospitalization_90d": 1,
                "recent_antibiotic_use_90d": 1,
                "num_prior_uti_1yr": 2,
                "catheter_use": 0,
                "immunocompromised": 1,
                "nursing_home_resident": 0,
                "prior_resistant_culture_1yr": 1,
                "creatinine_mg_dl": 1.95,
                "wbc_count_k_ul": 16.2,
                "travel_last_6mo": 1,
                "healthcare_worker": 0,
            }
        },
        {
            "id": "preset_pyelonephritis_young",
            "name": "Acute Pyelonephritis (Moderate Risk)",
            "description": "36yo female with fever, leukocytosis, and prior antibiotic use 2 months ago.",
            "data": {
                "age": 36,
                "sex": "F",
                "infection_type": "Pyelonephritis",
                "organism": "E_coli",
                "diabetes": 0,
                "recent_hospitalization_90d": 0,
                "recent_antibiotic_use_90d": 1,
                "num_prior_uti_1yr": 1,
                "catheter_use": 0,
                "immunocompromised": 0,
                "nursing_home_resident": 0,
                "prior_resistant_culture_1yr": 0,
                "creatinine_mg_dl": 1.05,
                "wbc_count_k_ul": 13.5,
                "travel_last_6mo": 0,
                "healthcare_worker": 1,
            }
        }
    ]

    return {
        "valid_categories": VALID_CATEGORIES,
        "numerical_bounds": NUMERICAL_BOUNDS,
        "feature_display_names": FEATURE_DISPLAY_NAMES,
        "antibiotics": list(ANTIBIOTIC_TARGET_MAP.keys()),
        "risk_thresholds": RISK_THRESHOLDS,
        "presets": presets,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
        "attribution_notice": MODEL_ATTRIBUTION_NOTICE,
    }


@app.post("/api/predict")
def api_predict_patient(payload: PatientInputSchema):
    """Executes multi-model AMR resistance probability prediction for a patient."""
    try:
        data_dict = payload.model_dump()
        result = predict_patient(data_dict)
        return result
    except DataValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/explain")
def api_explain_prediction(payload: ExplainRequestSchema):
    """Generates local SHAP feature attributions and waterfall decomposition."""
    try:
        data_dict = payload.patient_data.model_dump()
        result = explain_patient_prediction(data_dict, payload.antibiotic)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.get("/api/models")
def get_model_registry_and_metrics():
    """Returns model selection registry, candidate comparison table, and ROC/PR curves."""
    try:
        reg_file = ARTIFACTS_DIR / "model_registry.json"
        metrics_file = ARTIFACTS_DIR / "metrics.json"

        if not reg_file.exists() or not metrics_file.exists():
            raise HTTPException(status_code=404, detail="Model artifacts not found. Please train models first.")

        with open(reg_file, "r") as f:
            registry = json.load(f)

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        return {
            "registry": registry,
            "detailed_metrics": metrics,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eda")
def get_dataset_eda():
    """Returns dataset summary statistics and feature distributions."""
    try:
        eda_file = ARTIFACTS_DIR / "eda_summary.json"
        if not eda_file.exists():
            raise HTTPException(status_code=404, detail="EDA artifact not found.")

        with open(eda_file, "r") as f:
            eda_data = json.load(f)

        return eda_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/global-shap")
def get_global_shap(antibiotic: Optional[str] = None):
    """Returns dataset-wide SHAP feature importance rankings."""
    try:
        return get_global_shap_summary(antibiotic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
