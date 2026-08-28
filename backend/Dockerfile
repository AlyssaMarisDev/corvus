FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY src ./src

EXPOSE 4000

# Note: when running in a container, set OLLAMA_HOST=http://host.docker.internal:11434
# and DATABASE_URL to reach the Postgres container (e.g. via --network or host.docker.internal).
CMD ["node", "src/index.js"]
