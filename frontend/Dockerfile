# Runs the Vite dev server inside the container — matches this app's
# "local research/demo tool" framing (see docs/architecture.md), not a
# production deployment. --host is required so the dev server listens on
# all interfaces inside the container, not just 127.0.0.1 (otherwise the
# published port would connect but hang).
FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
