"""
Convenience entry point: run `python run.py` instead of typing out the
full uvicorn command. Equivalent to:
    uvicorn app.main:app --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
