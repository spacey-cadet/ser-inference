from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.calibration.calibrate import CalibrationMap
from app.config import get_settings
from app.logging_pipeline.feature_log import build_feature_log
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

    print(f"Calibration: {app.state.calibration.method}")
    print(f"Session backend: {settings.session_backend}")
    print(f"Feature log backend: {settings.feature_log_backend}")
    print(f"Canary: {settings.canary_pct}% -> {settings.challenger_model_id or 'none'}")

    yield
    # No teardown needed today; hf_dataset backend flush is triggered by a
    # scheduled job, not on shutdown, since free-tier Spaces can be killed
    # without running shutdown hooks.


app = FastAPI(title="SER Production Pipeline", lifespan=lifespan)
app.include_router(router)
