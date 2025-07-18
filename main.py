# # api/main.py
# import joblib
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from fastapi.openapi.docs import get_swagger_ui_html
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel, Field
# from swagger_ui_bundle import swagger_ui_4_path

# from utils.fe import FEATURES, transform

# app = FastAPI(
#     title="Mobile-Banking Fraud Detection API",
#     description="Scores transactions and returns risk probability + alert flag",
#     version="1.0.0",
# )

# # ---------- load model & threshold once ----------
# MODEL = joblib.load("fraud_model.pkl")
# BEST_THRESHOLD = 0.37          # <-- use your tuned value
# # -------------------------------------------------

# class Txn(BaseModel):
#     txn_id:      str
#     user_id:     str
#     amount:      float
#     txn_time:    str           # ISO datetime
#     txn_type:    str           # 'debit' / 'credit'
#     location_lat: float
#     location_long: float
#     total_txns:  int | None = Field(1, description="Running txn count for user")

# class TxnResponse(BaseModel):
#     txn_id:      str
#     fraud_prob:  float
#     is_fraud:    bool

# @app.post("/predict", response_model=TxnResponse)
# def predict(txn: Txn):
#     """Score a single transaction."""
#     try:
#         df_in = pd.DataFrame([txn.dict()])
#         X = transform(df_in)
#         prob = MODEL.predict_proba(X)[:, 1][0]
#         is_fraud = prob >= BEST_THRESHOLD
#         return TxnResponse(txn_id=txn.txn_id,
#                            fraud_prob=round(prob, 4),
#                            is_fraud=is_fraud)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
# # Serve Swagger UI from local static assets (no CDN)
# @app.get("/docs", include_in_schema=False)
# async def custom_swagger_html():
#     return get_swagger_ui_html(
#         openapi_url=app.openapi_url,
#         title="Mobile-Banking API Docs",
#         swagger_js_url="/static/swagger-ui-bundle.js",
#         swagger_css_url="/static/swagger-ui.css",
#     )

# # Mount local Swagger UI assets
# app.mount("/static", StaticFiles(directory=swagger_ui_4_path), name="static")
