# Behavior AI server deployment

This service is internal-only. It must listen on `127.0.0.1:8010`; browsers must
call the main CAPTCHA API, never this service.

## Where it runs

**Backend server `210.109.52.124`** (moved off the GPU server 2026-07-28).

It sat on the GPU box `61.109.239.231` for no reason other than free space —
nothing in the inference path touches a GPU:

- no `torch` / CUDA import anywhere under `app/`
- the model is a 1.1MB LightGBM bundle loaded with joblib
- `torch` was never even installed in the GPU deployment's venv
- measured footprint there: **RSS 165MB** on a 64GB / 16-core host

Running next to the backend also lets it reuse the backend's existing MySQL
connection instead of needing a DB account of its own.

## Database

The AI talks to `catchap_dev_db` on the **DB server `210.109.52.114:3306`**, not
to a local MySQL. The move does not change this; the DB was always remote.

> **Do NOT apply `db/schema_mysql.sql`.** It is deprecated and would collide with
> the deployed tables, which are a different design. The ORM in
> `app/database/mysql_models.py` is mapped to what is actually deployed.
> See `docs/DB_REQUEST_20260728.md`.

The account needs `SELECT, INSERT, UPDATE` on the seven `ai_*` tables. `SELECT`
on `ai_pointer_events` is not optional — replay detection re-reads stored
trajectories, and without it the DTW check silently reports "no replay" instead
of failing.

## Install

1. Install from **`requirements-serve.txt`**, not `requirements.txt`. The latter
   pulls torch / xgboost / matplotlib / anthropic, none of which serving imports.

   ```
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements-serve.txt
   ```

2. Copy `behavior-ai.env.example` to `/etc/catchap/behavior-ai.env`, fill in the
   real values, then `chmod 600`.

3. Adjust `WorkingDirectory`, `ExecStart` and `User`/`Group` in
   `behavior-ai.service` to match the backend's layout.

4. Install the unit and enable it:

   ```
   sudo cp behavior-ai.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now behavior-ai
   ```

   `enable` matters — the GPU deployment was started by hand and would not have
   survived a reboot.

5. Check health from the server:

   ```
   curl -s http://127.0.0.1:8010/health
   ```

   Expect `status: ok`, `mysql_connected: true`, `model_loaded: true`,
   `policy_mode: shadow`. If the first call reports `mysql_connected: false` and
   later calls report `true`, `cryptography` is missing — install it.

## Model

`PRODUCTION_MODEL_DIR` points at a **candidate** bundle: `models/production/` is
still empty and no formal promotion has been run
(`training/select_best_model.py`).

Keep it in the env file, never in a shell variable. On the GPU host the model
was selected by an env var on a hand-started process, so restarting it silently
dropped the model — `/health` went to `model_loaded: false` with
`feature_schema_version` falling back to 1.0, and nothing on disk recorded which
bundle had been live.

## Policy

`RISK_POLICY_MODE=shadow` here, and `BEHAVIOR_POLICY_MODE=shadow` on the CAPTCHA
side, both stay as they are. Do not switch to `active` until the false-positive
rate has been measured against real traffic — the model recommends `step_up` for
trajectories it scores as non-human, and enabling that unmeasured risks blocking
real users. `ai_shadow_outcomes` is the table that makes the measurement
possible.
