# Ядро + зібрана вітрина в одному образі: ядро й так уміє роздавати статику, тож окремий
# веб-сервер під фронт не потрібен, а спільне походження прибирає CORS як клас.
FROM node:22-slim AS web
WORKDIR /src
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages ./packages
COPY apps ./apps
RUN pnpm install --frozen-lockfile
# Порожній VITE_LIVE_URL = «те саме походження»: фронт стукає в те ядро, що його й віддало.
RUN cd apps/web && VITE_LIVE_URL=same-origin npx vite build

FROM python:3.12-slim AS core
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/sim/pyproject.toml services/sim/
COPY services/sim/ploshcha_sim services/sim/ploshcha_sim
COPY services/sim/evalkit services/sim/evalkit
COPY services/sim/scripts services/sim/scripts
COPY packages/fixtures packages/fixtures
COPY contracts contracts
RUN pip install -e services/sim
COPY --from=web /src/apps/web/dist apps/web/dist
# Дев-редактор розкладки не має їхати в прод: він без автентифікації.
RUN rm -f apps/web/dist/layout_editor.html
# Стан (черга, ухвали, чутки, памʼять села) — на томі, інакше кожен перезапуск стирає село.
VOLUME ["/app/data"]
EXPOSE 8765
# Бінд лишається 127.0.0.1 у самому коді; у контейнері назовні дивиться проксі.
CMD ["python", "services/sim/scripts/serve_ploshcha.py", "--port", "8765", \
     "--db", "/app/data/ploshcha.db", "--resume"]
