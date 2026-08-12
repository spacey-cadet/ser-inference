"""
Small server-rendered review UI for the DynamoDB/S3-backed queue.
"""
from html import escape
from urllib.parse import quote

import boto3
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/admin", tags=["admin"])

_secret_cache: str | None = None


def _get_admin_secret(request: Request) -> str:
    global _secret_cache
    if _secret_cache is None:
        param_name = request.app.state.settings.admin_ui_secret_param
        if not param_name:
            raise HTTPException(status_code=404, detail="admin UI is not configured")
        resp = boto3.client("ssm").get_parameter(Name=param_name, WithDecryption=True)
        _secret_cache = resp["Parameter"]["Value"]
    return _secret_cache


def _require_auth(request: Request, token: str = Query(...)):
    if token != _get_admin_secret(request):
        raise HTTPException(status_code=403, detail="bad token")


@router.get("/label", response_class=HTMLResponse)
def label_page(request: Request, _=Depends(_require_auth), token: str = Query(...)):
    review_queue = request.app.state.review_queue
    if review_queue is None or not hasattr(review_queue, "mark_labeled"):
        raise HTTPException(status_code=404, detail="review queue is not configured")

    items = review_queue.list_pending(limit=20)
    bucket = request.app.state.settings.audio_bucket
    s3 = boto3.client("s3")
    quoted_token = quote(token, safe="")

    rows = []
    for item in items:
        audio_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": item["s3_audio_key"]},
            ExpiresIn=3600,
        )
        prediction = item.get("prediction", {})
        label = escape(str(prediction.get("label", "?")))
        confidence = escape(str(item.get("confidence", "?")))
        request_id = escape(str(item["request_id"]))
        rows.append(f"""
        <div class="item" data-request-id="{request_id}">
          <p><b>{request_id}</b> model said <code>{label}</code> (confidence {confidence})</p>
          <audio controls src="{escape(audio_url)}"></audio>
          <form method="post" action="/admin/label?token={quoted_token}">
            <input type="hidden" name="request_id" value="{request_id}">
            <select name="label">
              <option>Neutral</option><option>Happy</option><option>Sad</option>
              <option>Angry</option><option>Fear</option><option>Disgust</option>
              <option>Surprise</option>
            </select>
            <button type="submit">Submit label</button>
          </form>
        </div>
        """)

    html = f"""
    <html><head><title>SER review queue</title>
    <style>
      body{{font-family:sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;line-height:1.4}}
      .item{{border:1px solid #ccc;padding:1rem;margin-bottom:1rem;border-radius:8px}}
      audio{{display:block;width:100%;margin:.75rem 0}}
      button,select{{font:inherit}}
    </style>
    </head><body>
    <h1>Pending review ({len(items)})</h1>
    {''.join(rows) if rows else '<p>Nothing pending.</p>'}
    </body></html>
    """
    return HTMLResponse(html)


@router.post("/label")
async def submit_label(request: Request, token: str = Query(...), _=Depends(_require_auth)):
    form = await request.form()
    request_id = str(form["request_id"])
    label = str(form["label"])

    request.app.state.review_queue.mark_labeled(request_id, label)
    return RedirectResponse(url=f"/admin/label?token={quote(token, safe='')}", status_code=303)
