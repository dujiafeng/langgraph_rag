@echo off
REM Start Agent Chat UI (requires pnpm or npm)
cd chat-ui
pnpm dev 2>nul || npm run dev
