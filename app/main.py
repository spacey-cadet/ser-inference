from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.admin_ui import router as admin_ui_router
from app.api.routes import router
from app.calibration.calibrate import CalibrationMap
from app.config import get_settings
from app.logging_pipeline.feature_log import build_feature_log
from app.logging_pipeline.review_queue import build_review_queue
from app.model.registry import ModelRegistry
from app.session.state import build_session_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.settings = settings
    app.state.registry = ModelRegistry.bootstrap(settings)
    app.state.calibration = CalibrationMap.load(settings.calibration_path)
    app.state.session_store = build_session_store(settings)
    app.state.feature_log = build_feature_log(settings)
    app.state.review_queue = build_review_queue(settings, app.state.feature_log)

    print(f"Calibration: {app.state.calibration.method}")
    print(f"Session backend: {settings.session_backend}")
    print(f"Feature log backend: {settings.feature_log_backend}")
    print(f"Storage backend: {settings.storage_backend}")
    print(f"Canary: {settings.canary_pct}% -> {settings.challenger_model_id or 'none'}")

    yield
    # No teardown needed today; hf_dataset backend flush is triggered by a
    # scheduled job, not on shutdown, since free-tier Spaces can be killed
    # without running shutdown hooks.


app = FastAPI(title="SER Production Pipeline", lifespan=lifespan)
app.include_router(router)
app.include_router(admin_ui_router)

# AWS Lambda entrypoint. Inert for local uvicorn/docker usage.
from mangum import Mangum

handler = Mangum(app)
