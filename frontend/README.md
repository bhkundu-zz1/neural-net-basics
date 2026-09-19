# Paper Trading Frontend

React (Vite) frontend for the paper-trading simulator. Talks to the FastAPI
backend in `../backend/` — see `../docs/architecture.md` for the full
system overview and setup steps.

## Development

```bash
npm install
npm run dev       # http://localhost:5173
npm run test      # Vitest component tests
npm run build     # production build to dist/
```

Copy `.env.example` to `.env` if the backend isn't running at the default
`http://localhost:8000`:
```
VITE_API_BASE_URL=http://localhost:8000
```
