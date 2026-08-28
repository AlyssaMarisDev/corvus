# Dockerfile for Corvus PostgreSQL database
#
# Starts a plain, empty Postgres 16 — the schema is no longer baked in here
# via a docker-entrypoint-initdb.d script. It's built entirely by Alembic
# (backend/alembic/versions/), the same migration path used against the
# cloud database: run `alembic upgrade head` from backend/ (or just start
# the backend container, whose own CMD does this automatically) once this
# container is up.
FROM pgvector/pgvector:pg16

# Set environment variables for database configuration
ENV POSTGRES_DB=corvus
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=root

# Expose PostgreSQL port
EXPOSE 4202

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres -d corvus || exit 1
