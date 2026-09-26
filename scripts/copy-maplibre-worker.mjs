import { copyFileSync, mkdirSync } from 'node:fs'

mkdirSync('public/maplibre', { recursive: true })
copyFileSync('node_modules/maplibre-gl/dist/maplibre-gl-worker.mjs', 'public/maplibre/maplibre-gl-worker.mjs')
copyFileSync('node_modules/maplibre-gl/dist/maplibre-gl-shared.mjs', 'public/maplibre/maplibre-gl-shared.mjs')
