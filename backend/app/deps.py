"""
FastAPI Depends() providers: the app-startup-loaded model singleton and the
CouchDB database handle. Both are stashed on app.state by main.py's lifespan
and exposed here so routers don't reach into request.app.state directly.
"""

from fastapi import Request


def get_model(request: Request):
    return request.app.state.net, request.app.state.checkpoint, request.app.state.weights_path


def get_db(request: Request):
    return request.app.state.db
