import { useEffect, useRef } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { Map as MapLibreMap, Marker } from 'maplibre-gl'
import { Crosshair, Layers3 } from 'lucide-react'
import { type SharedNetwork } from './timeline'
import 'maplibre-gl/dist/maplibre-gl.css'

maplibregl.setWorkerUrl('/maplibre/maplibre-gl-worker.mjs')

type Vehicle = {
  id: string; position: [number, number]; heading: number | null;
  estimateSeconds: number | null
}
type Status = 'normal' | 'minor' | 'delay' | 'critical' | 'early' | 'unknown'
type Props = {
  vehicles: Vehicle[]; selected: Vehicle | null; selectedId: string | null;
  selectedStatus: Status; onSelect: (id: string) => void;
  showRoutes: boolean; setShowRoutes: (value: boolean) => void;
  collapsed: boolean; network: SharedNetwork;
}

const statusColor: Record<Status, string> = {
  normal: '#16b77c', minor: '#e6a619', delay: '#fa7c3e',
  critical: '#ed4255', early: '#4188e9', unknown: '#0865ff',
}

function networkGeoJSON(network: SharedNetwork): GeoJSON.FeatureCollection<GeoJSON.LineString> {
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
  for (const [a, b, owners] of network.edges) if (owners.includes(vehicleId)) {
    bounds.extend([network.nodes[a][1], network.nodes[a][0]])
    bounds.extend([network.nodes[b][1], network.nodes[b][0]])
    count++
  }
  return count ? bounds : null
}

export default function TransportMap({ vehicles, selected, selectedId, selectedStatus, onSelect,
  showRoutes, setShowRoutes, collapsed, network }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const markers = useRef(new Map<string, Marker>())
  const latestSelect = useRef(onSelect)
  latestSelect.current = onSelect

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
      map.on('click', 'network', event => {
        const properties = event.features?.[0]?.properties
        if (!properties) return
        const owners: string[] = typeof properties.owners === 'string' ? JSON.parse(properties.owners) : properties.owners
        if (owners?.length) latestSelect.current(owners[0])
      })
      map.on('mouseenter', 'network', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', 'network', () => { map.getCanvas().style.cursor = '' })
    }
    map.on('load', installLayers)
    const resize = new ResizeObserver(() => map.resize())
    resize.observe(host.current)
    return () => {
      resize.disconnect()
      markers.current.forEach(marker => marker.remove())
      markers.current.clear()
      map.remove()
      mapRef.current = null
    }
  }, [network])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const visible = new Set(vehicles.map(vehicle => vehicle.id))
    markers.current.forEach((marker, id) => {
      if (!visible.has(id)) { marker.remove(); markers.current.delete(id) }
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
        marker = created
      }
      marker.setLngLat([vehicle.position[1], vehicle.position[0]])
      const element = marker.getElement()
      const seconds = vehicle.estimateSeconds
      const status = seconds === null ? 'unknown' : seconds < -120 ? 'early' : seconds <= 60 ? 'normal' : seconds <= 120 ? 'minor' : seconds <= 300 ? 'delay' : 'critical'
      element.classList.add('map-bus-marker')
      element.classList.remove('normal', 'minor', 'delay', 'critical', 'early', 'unknown', 'selected', 'dimmed')
      element.classList.add(status)
      if (selectedId === vehicle.id) element.classList.add('selected')
      else if (selectedId) element.classList.add('dimmed')
      element.style.setProperty('--heading', `${vehicle.heading ?? 0}deg`)
    }
  }, [vehicles, selectedId])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const update = () => {
      if (!map.getLayer('network')) return
      map.setLayoutProperty('network', 'visibility', showRoutes ? 'visible' : 'none')
      map.setLayoutProperty('selected-network-halo', 'visibility', showRoutes && selectedId ? 'visible' : 'none')
      map.setLayoutProperty('selected-network', 'visibility', showRoutes && selectedId ? 'visible' : 'none')
      map.setPaintProperty('network', 'line-opacity', selectedId ? 0.14 : 0.75)
      map.setPaintProperty('selected-network', 'line-color', statusColor[selectedStatus])
      const filter: maplibregl.FilterSpecification = selectedId
        ? ['in', selectedId, ['get', 'owners']]
        : ['==', ['get', 'selected'], true]
      map.setFilter('selected-network-halo', filter)
      map.setFilter('selected-network', filter)
    }
    if (map.isStyleLoaded()) update()
    else map.once('load', update)
  }, [selectedId, selectedStatus, showRoutes])

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
    <div className="map-tools">
      <button title="Показать или скрыть траектории" aria-label="Показать или скрыть траектории" className={showRoutes ? 'active' : ''} onClick={() => setShowRoutes(!showRoutes)}><Layers3 size={19} /></button>
      <button title="Вернуться к обзору Москвы" aria-label="Вернуться к обзору Москвы" onClick={() => onSelect('')}><Crosshair size={19} /></button>
    </div>
  </div>
}
