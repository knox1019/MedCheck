from fastapi import FastAPI

app = FastAPI(title="MedCheck", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "dev", "version": "0.1.0"}
