# Production Checklist

This checklist is tailored to the current `hr_helpdesk_rag` codebase and deployment model.

## 1. Environment and Secrets

- [ ] `APP_ENV=production` in production environment.
- [ ] `GOOGLE_API_KEY` is injected via secret manager (not committed in files).
- [ ] `APP_API_KEY` is set and rotated periodically.
- [ ] Database credentials are non-default and environment-specific.
- [ ] Redis credentials and TLS settings are configured if using managed Redis.

Recommended production values:

```env
APP_ENV=production
APP_API_KEY=<strong-random-key>
RATE_LIMIT_BACKEND=redis
RATE_LIMIT_REDIS_URL=redis://<redis-host>:6379/0
RATE_LIMIT_REDIS_PREFIX=prod:snailcloud:ratelimit
RATE_LIMIT_PER_MINUTE=60
API_WORKERS=2
```

## 2. Rate Limiting and Abuse Protection

- [ ] `RATE_LIMIT_BACKEND=redis` enabled for multi-instance consistency.
- [ ] `RATE_LIMIT_REDIS_URL` reachable from API container/host.
- [ ] `RATE_LIMIT_REDIS_PREFIX` namespaced per environment (`dev`, `staging`, `prod`).
- [ ] Alert configured for spikes in `429` responses.
- [ ] Alert configured for `503 Rate limiter unavailable` responses.

## 3. API Security

- [ ] TLS termination configured at ingress/load balancer.
- [ ] API is not exposed without auth in public environments.
- [ ] `X-Forwarded-For` is trusted only from known proxy/ingress.
- [ ] Request size and timeout limits are enforced at gateway/ingress.
- [ ] `/docs`, `/redoc`, `/openapi.json` are disabled (`APP_ENV=production`).

## 4. Logging and Observability

- [ ] `logs/api.log` and `logs/http.jsonl` are shipped to centralized logging.
- [ ] `X-Request-ID` is preserved across proxy and app logs.
- [ ] Dashboard includes:
  - [ ] p95/p99 `/ask` latency
  - [ ] 5xx error rate
  - [ ] 429 and 503 rate-limit counts
  - [ ] retrieval vs generation latency split
- [ ] Log retention and rotation policy is configured.

## 5. Privacy and Data Handling

- [ ] Feedback logging remains metadata-only (hash + length), no raw PII.
- [ ] Retention window for feedback logs is documented and enforced.
- [ ] Access to logs/storage is restricted by role.
- [ ] Backups are encrypted and access-audited.

## 6. Database and Vector Store

- [ ] PostgreSQL/pgvector backups are enabled and tested.
- [ ] Restore drill documented and periodically tested.
- [ ] Vector collection (`COLLECTION_NAME`) matches deployment environment.
- [ ] Health/latency monitoring on DB connections is enabled.

## 7. Quality and Release Gates

- [ ] CI passes:
  - [ ] `python -m pytest -q`
  - [ ] lint/type checks
  - [ ] dependency scan (`pip-audit` or equivalent)
- [ ] Pre-release RAG evaluation run:
  - [ ] `python 06_rag_evaluation.py`
  - [ ] `python scripts/check_quality_gate.py --summary rag_evaluation_summary.json`
- [ ] Metrics thresholds meet release policy.

## 8. Deployment and Runtime

- [ ] App is deployed with restart policy and health checks.
- [ ] `run_api_production.py` (or equivalent process manager command) is used.
- [ ] Container image is immutable and version-tagged.
- [ ] Rollback artifact for previous stable version is available.

## 9. Incident Readiness

- [ ] Runbook exists for:
  - [ ] Gemini API outage
  - [ ] Redis outage
  - [ ] PostgreSQL outage
  - [ ] Elevated 5xx/latency incidents
- [ ] Team uses `request_id` for trace-based incident triage.
- [ ] On-call escalation path is documented.

## 10. Post-Deploy Verification (Smoke Test)

Run after every production deploy:

1. `GET /health` returns `200` and expected model/collection metadata.
2. `POST /ask` with valid API key returns `200` and answer payload.
3. Burst `/ask` requests to confirm `429` behavior at configured threshold.
4. Temporarily block Redis (staging test) to verify graceful `503 Rate limiter unavailable` behavior.
5. Confirm request appears in structured HTTP logs with `request_id`.

## Optional Docker Compose Add-On (Redis Service)

If running compose in production-like environments, add a Redis service and point `RATE_LIMIT_REDIS_URL` to it, for example:

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: snailcloud-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
```

Then set:

```env
RATE_LIMIT_BACKEND=redis
RATE_LIMIT_REDIS_URL=redis://redis:6379/0
```
