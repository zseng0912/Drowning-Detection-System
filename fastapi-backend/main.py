"""
Drowning Detection System - FastAPI Backend Entry Point

This is the entry point for the FastAPI application.
It imports the app from the modular structure.
"""

from app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)