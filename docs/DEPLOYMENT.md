# Deployment Guide: E-Commerce Review Intelligence System

This document provides production deployment procedures for the **E-Commerce Review Intelligence System**, encompassing both the Python FastAPI machine learning microservice and the React/Tailwind frontend application.

---

## 1. Architecture Overview

```
                      +-----------------------------+
                      |   React SPA Frontend        |
                      |   (Vercel / Netlify / Vite) |
                      +--------------+--------------+
                                     |
                                     | HTTPS REST (JSON / Multipart)
                                     v
                      +-----------------------------+
                      |   FastAPI ML Microservice   |
                      |   (Render / Hugging Face)   |
                      |   Port 8000                 |
                      +--------------+--------------+
                                     |
                                     | In-Memory Inference (<2ms)
                                     v
                      +-----------------------------+
                      |   best_model.joblib         |
                      |   (Linear SVM + TF-IDF)     |
                      +-----------------------------+
```

---

## 2. Backend ML Service Deployment

### Option A: Render (Recommended - Free Web Service)

1. Sign up or log into [Render.com](https://render.com).
2. Click **New +** &rarr; **Web Service**.
3. Connect your GitHub repository: `[GITHUB_REPOSITORY_URL]`.
4. Configure service settings:
   - **Root Directory**: `ml-service`
   - **Environment**: `Docker` (or `Python 3`)
   - **Build Command** (if native Python): `pip install -r requirements.txt`
   - **Start Command** (if native Python): `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free ($0/month)
5. Set Environment Variables in Render Dashboard:
   | Key | Value | Note |
   |-----|-------|------|
   | `PORT` | `8000` | Render injects this automatically |
   | `CORS_ALLOWED_ORIGINS` | `*` or `https://your-frontend.vercel.app` | Comma-separated allowed origins |
6. Click **Deploy Web Service**.
7. Once deployed, note your service URL (e.g., `https://review-ml-api.onrender.com`).
8. Verify health endpoint: `curl https://review-ml-api.onrender.com/health`.

### Option B: Hugging Face Spaces (Docker Space)

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Docker** as the Space SDK.
3. Push the contents of `ml-service/` (including `Dockerfile`, `requirements.txt`, `models/best_model.joblib`).
4. Hugging Face Spaces automatically listens on port `7860`.
   - Set environment variable: `PORT=7860`.
5. Your endpoint will be live at `https://[username]-[space-name].hf.space`.

---

## 3. Frontend Deployment

### Option A: Vercel (Recommended)

1. Sign up or log into [Vercel](https://vercel.com).
2. Click **Add New...** &rarr; **Project** and import your repository.
3. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `./` (root)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Configure Environment Variables:
   | Key | Value | Description |
   |-----|-------|-------------|
   | `VITE_API_BASE_URL` | `https://review-ml-api.onrender.com` | URL of your deployed ML service (no trailing slash) |
5. Click **Deploy**.
6. Visit your live app URL (e.g., `https://review-intelligence.vercel.app`).

### Option B: Netlify

1. In [Netlify](https://netlify.com), click **Add new site** &rarr; **Import an existing project**.
2. Select your repository.
3. Build Settings:
   - **Base directory**: (empty / root)
   - **Build command**: `npm run build`
   - **Publish directory**: `dist`
4. Set Environment Variable: `VITE_API_BASE_URL=https://your-ml-api.onrender.com`.
5. Add a `public/_redirects` file with: `/* /index.html 200` to support client-side routing.
6. Click **Deploy Site**.

---

## 4. Free-Tier Cold Starts & Mitigation

- **Render / Hugging Face Free Tier**: Free tier backend instances automatically spin down after 15 minutes of inactivity. The first request after sleep can take 30–50 seconds to warm up.
- **Frontend Fallback**: The frontend application detects unreachable backends and gracefully renders a prominent **"Offline Snapshot"** badge, utilizing pre-computed empirical evaluation metrics while displaying clear instructions.
- **Keep-Alive Recommendation**: Use a free uptime monitor (e.g., [Cron-Job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com)) to ping `GET /health` every 10 minutes prior to a live presentation or grading review.

---

## 5. Local Presentation "Plan B" (Guaranteed 100% Demo Reliability)

For in-person grading presentations, internet or cloud platform latency can cause unexpected demo stalls. Use this local setup for 100% reliability:

```bash
# 1. Clone repository
git clone [GITHUB_REPOSITORY_URL]
cd review-intelligence

# 2. Run automated verification script (validates all tests and launches services)
bash scripts/verify_all.sh

# 3. In terminal 1: Launch FastAPI ML service
cd ml-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. In terminal 2: Launch Vite development server
npm install
npm run dev
```

Open `http://localhost:3000` in your browser. All API calls route seamlessly to `http://localhost:8000` through the built-in Vite reverse proxy.
