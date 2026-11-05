"""Server-rendered pages (Jinja2 + HTMX). Complete — no TODOs here."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.deps import Components, get_components, get_store
from app.services import UploadError, run_chat_turn, save_and_index_upload
from app.storage.json_store import SessionNotFoundError, SessionStore

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter(include_in_schema=False)


def _redirect(request: Request, url: str) -> Response:
    if request.headers.get("HX-Request"):
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def _render_page(request: Request, store: SessionStore, components: Components, session_id: str | None) -> HTMLResponse:
    current = None
    if session_id:
        try:
            current = store.get(session_id)
        except SessionNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sessions": store.list(),
            "current": current,
            "missing": components.missing,
        },
    )


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> Response:
    sessions = store.list()
    if sessions:
        return RedirectResponse(f"/sessions/{sessions[0].id}", status_code=status.HTTP_303_SEE_OTHER)
    return _render_page(request, store, components, None)


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def session_page(
    session_id: str,
    request: Request,
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    return _render_page(request, store, components, session_id)


@router.post("/ui/sessions")
def ui_create_session(request: Request, store: SessionStore = Depends(get_store)) -> Response:
    session = store.create()
    return _redirect(request, f"/sessions/{session.id}")


@router.post("/ui/sessions/{session_id}/rename")
def ui_rename_session(
    session_id: str,
    request: Request,
    title: str = Form(...),
    store: SessionStore = Depends(get_store),
) -> Response:
    try:
        store.rename(session_id, title)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return _redirect(request, f"/sessions/{session_id}")


@router.post("/ui/sessions/{session_id}/delete")
def ui_delete_session(session_id: str, request: Request, store: SessionStore = Depends(get_store)) -> Response:
    try:
        store.delete(session_id)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return _redirect(request, "/")


@router.post("/ui/sessions/{session_id}/messages", response_class=HTMLResponse)
def ui_send_message(
    session_id: str,
    request: Request,
    content: str = Form(..., min_length=1, max_length=8000),
    store: SessionStore = Depends(get_store),
    components: Components = Depends(get_components),
) -> HTMLResponse:
    try:
        user_message, assistant_message = run_chat_turn(store, components.engine, session_id, content)
    except SessionNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    return templates.TemplateResponse(
        request, "partials/messages.html", {"messages": [user_message, assistant_message]}
    )


@router.post("/ui/documents", response_class=HTMLResponse)
async def ui_upload_document(
    request: Request,
    file: UploadFile,
    components: Components = Depends(get_components),
) -> HTMLResponse:
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_upload_bytes + 1)
    try:
        result = save_and_index_upload(settings, components, file.filename or "", data)
        message, ok = f"{result.filename}: {result.detail}", True
    except UploadError as exc:
        message, ok = str(exc), False
    return templates.TemplateResponse(request, "partials/upload_status.html", {"message": message, "ok": ok})
