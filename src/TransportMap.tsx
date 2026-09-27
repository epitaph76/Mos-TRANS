import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { Map as MapLibreMap, Marker } from 'maplibre-gl'
import { Crosshair, Layers3 } from 'lucide-react'
import { snapToVehicleRoute, type SharedNetwork } from './timeline'
import { riskStatus } from './risk'
import 'maplibre-gl/dist/maplibre-gl.css'

maplibregl.setWorkerUrl('/maplibre/maplibre-gl-worker.mjs?v=6.11.2-js-mime')

type Vehicle = {
  id: string; position: [number, number]; heading: number | null;
  estimateSeconds: number | null; probability: number | null;
  forecastExpired?: boolean;
  gpsEventTime?: string | null;
  positionMethod?: 'gps' | 'route' | 'heading'
}
type Props = {
  vehicles: Vehicle[]; selected: Vehicle | null; selectedId: string | null;
  onSelect: (id: string) => void;
  onRouteSelect: (id: string) => void;
  showRoutes: boolean; setShowRoutes: (value: boolean) => void;
  collapsed: boolean; network: SharedNetwork;
  replayTime: number | null;
}

type Correction = { from: [number, number]; to: [number, number]; started: number }
const CORRECTION_MS = 900

function networkGeoJSON(network: SharedNetwork): GeoJSON.FeatureCollection<GeoJSON.LineString> {
  if (network.roadRoutes) {
    const owners = network.roadRoutes.segments.map(() => new Set<string>())
    for (const [vehicleId, sequence] of Object.entries(network.roadRoutes.vehicles))
      for (const segmentId of sequence) if (segmentId >= 0) owners[segmentId].add(vehicleId)
    return { type: 'FeatureCollection', features: network.roadRoutes.segments.map((points, index) => ({
      type: 'Feature', geometry: { type: 'LineString', coordinates: points.map(([lat, lon]) => [lon, lat]) },
      properties: { owners: [...owners[index]] },
    })) }
  }
  return { type: 'FeatureCollection', features: network.edges.map(([a, b, owners]) => ({
    type: 'Feature', geometry: { type: 'LineString', coordinates: [
      [network.nodes[a][1], network.nodes[a][0]],
      [network.nodes[b][1], network.nodes[b][0]],
    ] }, properties: { owners },
  })) }
}

function routeBounds(network: SharedNetwork, vehicleId: string) {
  const bounds = new maplibregl.LngLatBounds()
  let count = 0
  if (network.roadRoutes?.vehicles[vehicleId]) {
    for (const segmentId of network.roadRoutes.vehicles[vehicleId]) {
      if (segmentId < 0) continue
      for (const [lat, lon] of network.roadRoutes.segments[segmentId]) {
        bounds.extend([lon, lat]); count++
      }
    }
    return count ? bounds : null
  }
  for (const [a, b, owners] of network.edges) if (owners.includes(vehicleId)) {
    bounds.extend([network.nodes[a][1], network.nodes[a][0]])
    bounds.extend([network.nodes[b][1], network.nodes[b][0]])
    count++
  }
  return count ? bounds : null
}

export default function TransportMap({ vehicles, selected, selectedId, onSelect, onRouteSelect,
  showRoutes, setShowRoutes, collapsed, network, replayTime }: Props) {
  const [showHint, setShowHint] = useState(() => {
    try { return window.localStorage.getItem('mostrans-route-hint-seen') !== '1' }
    catch { return true }
  })
  useEffect(() => {
    if (!showHint) return
    try { window.localStorage.setItem('mostrans-route-hint-seen', '1') } catch { /* storage may be unavailable */ }
    const timer = window.setTimeout(() => setShowHint(false), 6000)
    return () => window.clearTimeout(timer)
  }, [showHint])
  const host = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const markers = useRef(new Map<string, Marker>())
  const corrections = useRef(new Map<string, Correction>())
  const gpsKeys = useRef(new Map<string, string | null>())
  const animationFrame = useRef<number | null>(null)
  const lastReplayTime = useRef<number | null>(null)
  const latestSelect = useRef(onSelect)
  latestSelect.current = onSelect
  const latestRouteSelect = useRef(onRouteSelect)
  latestRouteSelect.current = onRouteSelect

  const animateCorrections = () => {
    animationFrame.current = null
    const now = performance.now()
    corrections.current.forEach((correction, id) => {
      const marker = markers.current.get(id)
      if (!marker) { corrections.current.delete(id); return }
      const progress = Math.min(1, (now - correction.started) / CORRECTION_MS)
      const eased = progress * progress * (3 - 2 * progress)
      const interpolated: [number, number] = [
        correction.from[0] + (correction.to[0] - correction.from[0]) * eased,
        correction.from[1] + (correction.to[1] - correction.from[1]) * eased,
      ]
      const [lat, lon] = snapToVehicleRoute([interpolated[1], interpolated[0]], network, id)
      marker.setLngLat([lon, lat])
      if (progress === 1) corrections.current.delete(id)
    })
    if (corrections.current.size) animationFrame.current = requestAnimationFrame(animateCorrections)
  }

  useEffect(() => {
    if (!host.current) return
    const map = new maplibregl.Map({
      container: host.current, center: [37.617, 55.752], zoom: 10,
      minZoom: 3, maxZoom: 19, dragRotate: false,
      style: 'https://tiles.openfreemap.org/styles/bright',
      attributionControl: { compact: true },
    })
    mapRef.current = map
    const geojson = networkGeoJSON(network)
    const installLayers = () => {
      if (map.getSource('transport-network')) return
      map.addSource('transport-network', { type: 'geojson', data: geojson })
      map.addLayer({ id: 'network', type: 'line', source: 'transport-network', paint: {
        'line-color': '#1182d3', 'line-width': 2.5, 'line-opacity': 0.75,
      }, layout: { 'line-cap': 'round', 'line-join': 'round' } })
      map.addLayer({ id: 'selected-network-halo', type: 'line', source: 'transport-network',
        filter: ['==', ['get', 'selected'], true],
        paint: { 'line-color': '#fff', 'line-width': 7, 'line-opacity': 0.9 },
        layout: { 'line-cap': 'round', 'line-join': 'round' } })
      map.addLayer({ id: 'selected-network', type: 'line', source: 'transport-network',
        filter: ['==', ['get', 'selected'], true],
        paint: { 'line-color': '#0865ff', 'line-width': 4 },
        layout: { 'line-cap': 'round', 'line-join': 'round' } })
      const selectNetwork = (event: maplibregl.MapLayerMouseEvent) => {
        const properties = event.features?.[0]?.properties
        if (!properties) return
        const owners: string[] = typeof properties.owners === 'string' ? JSON.parse(properties.owners) : properties.owners
        if (owners?.length) { setShowHint(false); latestRouteSelect.current(owners[0]) }
      }
      map.on('click', 'network', selectNetwork)
      map.on('click', 'selected-network', selectNetwork)
      map.on('mouseenter', 'network', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', 'network', () => { map.getCanvas().style.cursor = '' })
      map.on('mouseenter', 'selected-network', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', 'selected-network', () => { map.getCanvas().style.cursor = '' })
    }
    map.on('load', installLayers)
    const resize = new ResizeObserver(() => map.resize())
    resize.observe(host.current)
    return () => {
      resize.disconnect()
      if (animationFrame.current !== null) cancelAnimationFrame(animationFrame.current)
      animationFrame.current = null
      corrections.current.clear()
      gpsKeys.current.clear()
      lastReplayTime.current = null
      markers.current.forEach(marker => marker.remove())
      markers.current.clear()
      map.remove()
      mapRef.current = null
    }
  }, [network])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const timeJump = replayTime !== null && lastReplayTime.current !== null &&
      (replayTime < lastReplayTime.current - 0.5 || replayTime - lastReplayTime.current > 30)
    lastReplayTime.current = replayTime
    const visible = new Set(vehicles.map(vehicle => vehicle.id))
    markers.current.forEach((marker, id) => {
      if (!visible.has(id)) {
        marker.remove(); markers.current.delete(id)
        corrections.current.delete(id); gpsKeys.current.delete(id)
      }
    })
    for (const vehicle of vehicles) {
      let marker = markers.current.get(vehicle.id)
      if (!marker) {
        const element = document.createElement('button')
        element.type = 'button'
        element.className = 'map-bus-marker'
        element.title = `ТС ${vehicle.id}`
        element.setAttribute('aria-label', `ТС ${vehicle.id}`)
        element.innerHTML = '<span class="vehicle-capsule"><span class="capsule-windscreen"></span><span class="capsule-roof"></span><span class="capsule-tail"></span></span>'
        element.addEventListener('click', event => { event.stopPropagation(); latestSelect.current(vehicle.id) })
        const created = new maplibregl.Marker({ element, anchor: 'center', subpixelPositioning: true })
          .setLngLat([vehicle.position[1], vehicle.position[0]]).addTo(map)
        markers.current.set(vehicle.id, created)
        gpsKeys.current.set(vehicle.id, vehicle.gpsEventTime ?? null)
        marker = created
      }
      const target: [number, number] = [vehicle.position[1], vehicle.position[0]]
      const previousGps = gpsKeys.current.get(vehicle.id)
      const currentGps = vehicle.gpsEventTime ?? null
      if (!timeJump && previousGps && currentGps && previousGps !== currentGps) {
        const current = marker.getLngLat()
        corrections.current.set(vehicle.id, { from: [current.lng, current.lat], to: target, started: performance.now() })
        if (animationFrame.current === null) animationFrame.current = requestAnimationFrame(animateCorrections)
      } else if (corrections.current.has(vehicle.id) && !timeJump) {
        corrections.current.get(vehicle.id)!.to = target
      } else {
        corrections.current.delete(vehicle.id)
        marker.setLngLat(target)
      }
      gpsKeys.current.set(vehicle.id, currentGps)
      const element = marker.getElement()
      element.title = `ТС ${vehicle.id} · ${vehicle.positionMethod === 'route' ? 'оценка движения по маршруту' : vehicle.positionMethod === 'heading' ? 'оценка движения по курсу' : 'полученный GPS'}`
      const status = riskStatus(vehicle.forecastExpired ? null : vehicle.probability)
      element.classList.add('map-bus-marker')
      element.classList.remove('normal', 'minor', 'delay', 'critical', 'early', 'unknown', 'selected', 'dimmed')
      element.classList.add(status)
      if (selectedId === vehicle.id) element.classList.add('selected')
      else if (selectedId) element.classList.add('dimmed')
      element.style.setProperty('--heading', `${vehicle.heading ?? 0}deg`)
    }
  }, [vehicles, selectedId, replayTime])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const update = () => {
      if (!map.getLayer('network')) return
      map.setLayoutProperty('network', 'visibility', showRoutes ? 'visible' : 'none')
      map.setLayoutProperty('selected-network-halo', 'visibility', showRoutes && selectedId ? 'visible' : 'none')
      map.setLayoutProperty('selected-network', 'visibility', showRoutes && selectedId ? 'visible' : 'none')
      map.setPaintProperty('network', 'line-opacity', selectedId ? 0.14 : 0.75)
      const filter: maplibregl.FilterSpecification = selectedId
        ? ['in', selectedId, ['get', 'owners']]
        : ['==', ['get', 'selected'], true]
      map.setFilter('selected-network-halo', filter)
      map.setFilter('selected-network', filter)
    }
    if (map.isStyleLoaded()) update()
    else map.once('load', update)
  }, [selectedId, showRoutes])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (selectedId && selected) map.flyTo({ center: [selected.position[1], selected.position[0]], zoom: Math.max(map.getZoom(), 12), duration: 650 })
    else if (selectedId) {
      const bounds = routeBounds(network, selectedId)
      if (bounds) map.fitBounds(bounds, { padding: 40, maxZoom: 12, duration: 650 })
    } else map.flyTo({ center: [37.617, 55.752], zoom: 10, duration: 650 })
  }, [selectedId, network])

  useEffect(() => { mapRef.current?.resize() }, [collapsed])

  return <div className="map-wrap">
    <div ref={host} className="vector-map" aria-label="Карта транспорта Москвы" />
    {showRoutes && showHint && <div className="map-route-hint">Нажмите на линию маршрута, чтобы добавить автобус</div>}
    <div className="map-tools">
      <button title="Показать или скрыть траектории" aria-label="Показать или скрыть траектории" className={showRoutes ? 'active' : ''} onClick={() => setShowRoutes(!showRoutes)}><Layers3 size={19} /></button>
      <button title="Вернуться к обзору Москвы" aria-label="Вернуться к обзору Москвы" onClick={() => onSelect('')}><Crosshair size={19} /></button>
    </div>
  </div>
}
