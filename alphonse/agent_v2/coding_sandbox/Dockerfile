FROM node:22-slim@sha256:5ec23b7c622b00547753bc6ffd78085dd868a68aeb41c8e0a1418f033e13d96d
RUN apt-get update && apt-get install -y --no-install-recommends python3 bubblewrap netcat-openbsd ca-certificates && rm -rf /var/lib/apt/lists/*
RUN npm --fetch-retries=0 --fetch-timeout=10000 install -g @openai/codex@0.156.1
RUN ln -s "$(find /usr/local/lib/node_modules/@openai/codex -path '*/bin/codex' -type f | head -1)" /usr/local/bin/codex-linux-sandbox
