#!/usr/bin/env python3
"""
Trading Bot Marketplace - Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
import uvicorn
from dotenv import load_dotenv
import os
import json
import logging
import importlib
import traceback

# Load .env so DEVELOPMENT_MODE is available for dev_sandbox
load_dotenv('.env')

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Env flags
DEVELOPMENT_MODE = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'

# Import from new structure
from core.database import engine
from core.models import Base
from api.endpoints import auth, bots, subscriptions, admin

# Try to import optional routers
exchanges = None
pricing = None
subscriptions_simple = None
try:
    from api.endpoints import exchanges as exchanges  # type: ignore
except Exception:
    exchanges = None
try:
    from api.endpoints import pricing as pricing  # type: ignore
except Exception:
    pricing = None
if DEVELOPMENT_MODE:
    try:
        from api.endpoints import subscriptions_simple as subscriptions_simple  # type: ignore
    except Exception:
        subscriptions_simple = None

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="Trading Bot Marketplace",
    description="A comprehensive marketplace for trading bot rental",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for sandbox UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Favicon to avoid 404s
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Include routers (always-on)
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(bots.router, prefix="/bots", tags=["Bots"])
app.include_router(subscriptions.router, prefix="/subscriptions", tags=["Subscriptions"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

# Include env-dependent routers
if DEVELOPMENT_MODE:
    if subscriptions_simple is not None:
        app.include_router(subscriptions_simple.router, prefix="/subscriptions-simple", tags=["Subscriptions (Simplified)"])
else:
    if exchanges is not None:
        app.include_router(exchanges.router, prefix="/exchanges", tags=["Exchanges"])
    if pricing is not None:
        app.include_router(pricing.router, prefix="/pricing", tags=["Pricing"])

# Include dev sandbox if available
DEV_SANDBOX_INCLUDED = False
try:
    mod = importlib.import_module("api.endpoints.dev_sandbox")
    app.include_router(mod.router, prefix="/api/dev-sandbox", tags=["Developer Sandbox"])
    DEV_SANDBOX_INCLUDED = True
    logger.info("Developer Sandbox router included at /api/dev-sandbox")
except Exception as e:
    logger.error(f"Failed to include Developer Sandbox: {e}")
    logger.error(traceback.format_exc())

@app.get("/dev-sb-import-check")
async def dev_sb_import_check():
    try:
        mod = importlib.import_module("api.endpoints.dev_sandbox")
        routes = []
        try:
            for r in getattr(mod, 'router').routes:
                routes.append(getattr(r, 'path', str(r)))
        except Exception:
            pass
        return {"ok": True, "routes": routes}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

@app.get("/")
async def root():
    return {
        "message": "Trading Bot Marketplace API",
        "version": "2.0.0",
        "development_mode": DEVELOPMENT_MODE,
        "dev_sandbox": DEV_SANDBOX_INCLUDED,
        "endpoints": {
            "auth": "/auth",
            "bots": "/bots",
            "admin": "/admin",
            "subscriptions": "/subscriptions",
            "subscriptions_simple": "/subscriptions-simple" if DEVELOPMENT_MODE and subscriptions_simple else None,
            "exchanges": "/exchanges" if (not DEVELOPMENT_MODE and exchanges) else None,
            "pricing": "/pricing" if (not DEVELOPMENT_MODE and pricing) else None,
            "dev_sandbox": "/api/dev-sandbox" if DEV_SANDBOX_INCLUDED else None,
            "static": "/static",
            "docs": "/docs",
            "health": "/health",
            "test_static": "/test-static",
            "routes": "/routes"
        }
    }

@app.get("/routes")
async def list_routes():
    paths = []
    for route in app.routes:
        try:
            paths.append(getattr(route, 'path'))
        except Exception:
            continue
    return {"count": len(paths), "paths": paths}

@app.get("/api/dev-sandbox/_health")
async def dev_sandbox_health():
    return {"included": DEV_SANDBOX_INCLUDED}

@app.get("/test-static")
async def test_static():
    import os
    static_dir = "static"
    try:
        files = os.listdir(static_dir) if os.path.exists(static_dir) else []
    except Exception:
        files = []
    return {
        "static_directory": os.path.abspath(static_dir),
        "exists": os.path.exists(static_dir),
        "html_files": [f for f in files if f.endswith('.html')]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "development_mode": DEVELOPMENT_MODE, "dev_sandbox": DEV_SANDBOX_INCLUDED}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
