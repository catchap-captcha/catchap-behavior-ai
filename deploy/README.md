# Behavior AI server deployment

This service is internal-only. It must listen on `127.0.0.1:8010`; browsers
must call the main CAPTCHA API, never this service.

Before starting it, have the database administrator apply
`db/schema_mysql.sql` and `db/migrations/20260723_shadow_mode.sql` when the
schema already exists.

1. Copy `behavior-ai.env.example` to `/etc/catchap/behavior-ai.env`, fill the
   real MySQL values and backend key, then set mode `600`.
2. Update the two `/srv/catchap/ai-service` paths and `User`/`Group` in
   `behavior-ai.service` if the server uses a different service account.
3. Install the unit, reload systemd, and start it.
4. From the server, request `http://127.0.0.1:8010/health`.

The health response must have `status: ok`, `model_loaded: true`, and
`policy_mode: shadow` before the main CAPTCHA API declares itself ready.

The candidate bundle in the environment example is the current two-view model.
It remains shadow-only because the latest VAE red-team pool did not hold a
stable sub-5% evasion rate.
