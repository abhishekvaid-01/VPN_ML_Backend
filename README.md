# VPN ML Backend

Flask API for VPN / Non-VPN prediction.

## Endpoints

- `GET /api/health`
- `GET /api/features`
- `POST /api/predict`
- `GET /api/plots`
- `GET /api/plots/<filename>`

## Local Run

```powershell
pip install -r requirements.txt
python app.py
```

## Deployment (Render/Railway)

- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

## Environment Variables

- `PORT` (provided by platform)
- `MODEL_PATH` (optional, defaults to `../artifacts/vpn_model_bundle.joblib` from this folder)
- `CORS_ORIGINS` (optional, comma-separated list, defaults to `*`)

Example:

```bash
CORS_ORIGINS=https://your-frontend.vercel.app
```
