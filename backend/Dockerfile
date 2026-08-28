FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY src ./src

EXPOSE 4000

# Note: when running in a container, set GEMINI_API_KEY (and optionally
# GEMINI_MODEL), and point DATABASE_URL at the Postgres container
# (e.g. via --network or host.docker.internal).
CMD ["node", "src/index.js"]
