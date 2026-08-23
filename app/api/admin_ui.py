"""
app/api/admin_ui.py 

Adds one server-rendered HTML page to the existing FastAPI app for the
solo human labeler: list pending review-queue items, play the audio via
a presigned S3 URL, submit a label. No separate frontend build or host
— it's mounted into the same app that already serves /predict.

Wire this in by including its router in main.py:

    from app.api.admin_ui import router as admin_ui_router
    app.include_router(admin_ui_router)

Protected by a single shared secret (checked via a query param or
cookie) pulled from SSM at cold start — fine for a one-person labeling
workflow, not meant to scale past that.
"""
import os

import boto3
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/admin", tags=["admin"])

_ssm = boto3.client("ssm")
_s3 = boto3.client("s3")

_secret_cache: str | None = None


def _get_admin_secret() -> str:
    global _secret_cache
    if _secret_cache is None:
        param_name = os.environ["ADMIN_UI_SECRET_PARAM"]
        resp = _ssm.get_parameter(Name=param_name, WithDecryption=True)
        _secret_cache = resp["Parameter"]["Value"]
    return _secret_cache


def _require_auth(token: str = Query(...)):
    if token != _get_admin_secret():
        raise HTTPException(status_code=403, detail="bad token")


@router.get("/label", response_class=HTMLResponse)
def label_page(request: Request, _=Depends(_require_auth), token: str = Query(...)):
    # NOTE: this calls into the existing review-queue backend's
    # list_pending() — import path below assumes the factory function
    # from config_additions.py's wiring. Adjust to match however
    # review_queue.py actually exposes the active backend instance.
    from app.logging_pipeline.review_queue import get_review_queue_backend

    backend = get_review_queue_backend()
    items = backend.list_pending(limit=20)
    bucket = os.environ["AUDIO_BUCKET"]

    rows = []
    for item in items:
        audio_url = _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": item["s3_audio_key"]},
            ExpiresIn=3600,
        )
        rows.append(f"""
        <div class="item" data-request-id="{item['request_id']}">
          <p><b>{item['request_id']}</b> — model said
             <code>{item['prediction'].get('label', '?')}</code>
             (confidence {item.get('confidence', '?')})</p>
          <audio controls src="{audio_url}"></audio>
          <form method="post" action="/admin/label?token={token}">
            <input type="hidden" name="request_id" value="{item['request_id']}">
            <select name="label">
              <option value="angry">Angry</option>
              <option value="calm">Calm</option>
              <option value="disgust">Disgust</option>
              <option value="fearful">Fearful</option>
              <option value="happy">Happy</option>
              <option value="neutral" selected>Neutral</option>
              <option value="sad">Sad</option>
              <option value="surprised">Surprised</option>
            </select>
            <!-- option VALUES must match scripts/eval_report.py's
                 IDX_TO_EMOTION values exactly (lowercase, includes
                 "calm", "fearful"/"surprised" not "fear"/"surprise").
                 kaggle/train_head_kernel.py's encode_label() will raise
                 loudly on anything else, rather than silently corrupt
                 a training batch — but better to never submit a bad
                 value in the first place. -->

            <button type="submit">Submit label</button>
          </form>
        </div>
        """)

    html = f"""
    <html><head><title>SER review queue</title>
    <style>body{{font-family:sans-serif;max-width:600px;margin:2rem auto}}
    .item{{border:1px solid #ccc;padding:1rem;margin-bottom:1rem;border-radius:8px}}</style>
    </head><body>
    <h1>Pending review ({len(items)})</h1>
    {''.join(rows) if rows else '<p>Nothing pending.</p>'}
    </body></html>
    """
    return HTMLResponse(html)


@router.post("/label")
async def submit_label(
    request: Request, token: str = Query(...), _=Depends(_require_auth)
):
    from app.logging_pipeline.review_queue import get_review_queue_backend

    form = await request.form()
    request_id = form["request_id"]
    label = form["label"]

    backend = get_review_queue_backend()
    backend.mark_labeled(request_id, label)

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/admin/label?token={token}", status_code=303)