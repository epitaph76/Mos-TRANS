# Mos-TRANS: серверная конфигурация

Публичный адрес: https://мостранс.хелпи.рф/

ASCII/Punycode: `xn--80axeckfde.xn--e1agiq1a.xn--p1ai`

TCP NDTP: `xn--80axeckfde.xn--e1agiq1a.xn--p1ai:9201`

Исходники на сервере: `/opt/mos-trans`. Три контейнера `ml`, `backend`, `frontend` запускаются:

```sh
cd /opt/mos-trans
docker compose -p mostrans -f compose.server.yaml up -d --no-build
docker compose -p mostrans -f compose.server.yaml ps
```

`frontend` доступен на 8080 и через Caddy/HTTPS. `backend` и `ml` по HTTP находятся во внутренней сети Docker; TCP-порт 9201 backend опубликован для терминалов NDTP. `compose.server.yaml` подключает frontend к существующей на этом сервере сети Caddy `dobrie-dela-max_default`; для другого хоста используйте обычный `docker-compose.yml` и настройте свой reverse proxy.

Пути сайта: `/` — живой дашборд, `/instructions` — проверка для жюри, `/docs` — документация, `/api` — Swagger, `/api/openapi.json` — схема, `/api/health` — состояние. Домен с кириллицей для TCP-клиентов рекомендуется передавать в виде Punycode.

Архив организаторов хранится в `data/dataset.zip` и не включён в Git или Docker-образы. Подготовленные данные находятся в `data/cache`. Frontend собирается локально `npm ci && npm run build`; затем `dist` переносится на сервер. `deploy/Dockerfile.frontend.server` копирует готовый `dist`. Backend запускается из `Dockerfile.backend`.

На сервере мало свободного места (при последней проверке около 490 МиБ); перед полной пересборкой Python-образа проверьте `df -h`. Краткая инструкция для проверки продукта: [instructions](https://мостранс.хелпи.рф/instructions).
