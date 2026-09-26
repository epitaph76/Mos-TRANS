export type TrackPoint = [number, number, number, number, number | null]
export type TimelineStop = { id: string; time: string; lon: number; lat: number; name: string }
export type TimelineForecast = {
  time: number; estimateSeconds: number; forecastTime: string;
  forecastStopId: string; sampleId: string
}
export type TimelineVehicle = {
  id: string; track: TrackPoint[];
  stops: TimelineStop[]; forecasts: TimelineForecast[]
}
export type SharedNetwork = {
  mergeMeters: number; nodes: [number, number][]; edges: [number, number, string[]][]
}

export function indexNetwork(network: SharedNetwork) {
  const cells = new Map<string, number[]>()
  const resolution = 0.001
  network.edges.forEach(([a, b], index) => {
    const from = network.nodes[a], to = network.nodes[b]
    const south = Math.floor((Math.min(from[0], to[0]) - 0.0005) / resolution)
    const north = Math.floor((Math.max(from[0], to[0]) + 0.0005) / resolution)
    const west = Math.floor((Math.min(from[1], to[1]) - 0.0008) / resolution)
    const east = Math.floor((Math.max(from[1], to[1]) + 0.0008) / resolution)
    for (let lat = south; lat <= north; lat++) for (let lon = west; lon <= east; lon++) {
      const key = `${lat}:${lon}`
      const bucket = cells.get(key) ?? []
      bucket.push(index)
      cells.set(key, bucket)
    }
  })
  return { network, cells, resolution }
}

export function snapToNetwork(position: [number, number], heading: number | null, indexed: ReturnType<typeof indexNetwork>, vehicleId: string): [number, number] {
  const [lat, lon] = position
  const { network, cells, resolution } = indexed
  let closest: [number, number] = position
  let best = 45
  const cosLat = Math.cos(lat * Math.PI / 180)
  const candidates = cells.get(`${Math.floor(lat / resolution)}:${Math.floor(lon / resolution)}`) ?? []
  for (const index of candidates) {
    const [a, b, owners] = network.edges[index]
    if (!owners.includes(vehicleId)) continue
    const from = network.nodes[a], to = network.nodes[b]
    const ax = (from[1] - lon) * 111320 * cosLat, ay = (from[0] - lat) * 111320
    const dx = (to[1] - from[1]) * 111320 * cosLat, dy = (to[0] - from[0]) * 111320
    const length2 = dx * dx + dy * dy
    if (length2 < 1) continue
    const fraction = Math.max(0, Math.min(1, -(ax * dx + ay * dy) / length2))
    const x = ax + dx * fraction, y = ay + dy * fraction
    const distance = Math.hypot(x, y)
    if (distance >= best) continue
    if (heading !== null) {
      const bearing = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360
      const delta = Math.abs((bearing - heading + 180) % 360 - 180)
      if (Math.min(delta, 180 - delta) > 65) continue
    }
    best = distance
    closest = [from[0] + (to[0] - from[0]) * fraction, from[1] + (to[1] - from[1]) * fraction]
  }
  return closest
}

export function formatTime(seconds: number, includeSeconds = false) {
  const whole = Math.floor(seconds)
  const hours = String(Math.floor(whole / 3600)).padStart(2, '0')
  const minutes = String(Math.floor((whole % 3600) / 60)).padStart(2, '0')
  return includeSeconds ? `${hours}:${minutes}:${String(whole % 60).padStart(2, '0')}` : `${hours}:${minutes}`
}

export function stopSeconds(value: string, day: string) {
  const hours = Number(value.slice(11, 13))
  const minutes = Number(value.slice(14, 16))
  const seconds = Number(value.slice(17, 19))
  return (value.slice(0, 10) === day ? 0 : 86400) + hours * 3600 + minutes * 60 + seconds
}

export function forecastAt(points: TimelineForecast[], time: number, day: string) {
  return points.find(point => {
    const horizon = stopSeconds(point.forecastTime, day) - time
    return point.time <= time && time - point.time < 60 && horizon > 600 && horizon <= 900
  })
}

export function precedingIndex(track: TrackPoint[], time: number) {
  let low = 0
  let high = track.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (track[middle][0] <= time) low = middle + 1
    else high = middle
  }
  return low - 1
}

export function positionAt(track: TrackPoint[], time: number) {
  const index = precedingIndex(track, time)
  if (index < 0) return null
  const previous = track[index]
  const age = time - previous[0]
  if (age > 180) return null
  return {
    position: [previous[1], previous[2]] as [number, number],
    point: previous,
    age,
    heading: previous[4],
  }
}

type RouteEdge = [number, number]
const routeEdgesCache = new WeakMap<SharedNetwork, Map<string, RouteEdge[]>>()

function routeEdges(network: SharedNetwork, vehicleId: string): RouteEdge[] {
  let byVehicle = routeEdgesCache.get(network)
  if (!byVehicle) { byVehicle = new Map(); routeEdgesCache.set(network, byVehicle) }
  let edges = byVehicle.get(vehicleId)
  if (!edges) {
    edges = network.edges.filter(([, , owners]) => owners.includes(vehicleId)).map(([a, b]) => [a, b])
    byVehicle.set(vehicleId, edges)
  }
  return edges
}

export type EstimatedPosition = { position: [number, number]; method: 'gps' | 'route' | 'heading' }

/** Move from an already received fix; the next GPS packet is never an input. */
export function estimateBetweenFixes(position: [number, number], speedKmh: number, heading: number | null,
  secondsSinceFix: number, network: SharedNetwork, vehicleId: string): EstimatedPosition {
  const observed: EstimatedPosition = { position, method: 'gps' }
  if (!Number.isFinite(secondsSinceFix) || secondsSinceFix <= 0 || !Number.isFinite(speedKmh) || speedKmh <= 0) return observed
  const elapsed = Math.min(secondsSinceFix, 30)
  const travel = Math.min(600, Math.min(speedKmh, 100) * elapsed / 3.6)
  const metersLat = 111_320
  const metersLon = metersLat * Math.cos(position[0] * Math.PI / 180)
  const edges = routeEdges(network, vehicleId)
  let best = { score: Infinity, distance: Infinity, index: -1, fraction: 0 }
  for (let index = 0; index < edges.length; index++) {
    const [a, b] = edges[index]
    const from = network.nodes[a], to = network.nodes[b]
    const dx = (to[1] - from[1]) * metersLon, dy = (to[0] - from[0]) * metersLat
    const length2 = dx * dx + dy * dy
    if (length2 < 1 || length2 > 4_000_000) continue
    const px = (position[1] - from[1]) * metersLon, py = (position[0] - from[0]) * metersLat
    const fraction = Math.max(0, Math.min(1, (px * dx + py * dy) / length2))
    const distance = Math.hypot(px - fraction * dx, py - fraction * dy)
    const bearing = (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360
    const headingDelta = heading === null ? 0 : Math.abs((bearing - heading + 540) % 360 - 180)
    const score = distance + (headingDelta > 100 ? 120 : headingDelta * 0.15)
    if (score < best.score) best = { score, distance, index, fraction }
  }
  if (best.index < 0 || best.distance > 300) {
    if (heading === null || !Number.isFinite(heading)) return observed
    const radians = heading * Math.PI / 180
    return { position: [position[0] + Math.cos(radians) * travel / metersLat,
      position[1] + Math.sin(radians) * travel / metersLon], method: 'heading' }
  }

  const [startA, startB] = edges[best.index]
  const startFrom = network.nodes[startA], startTo = network.nodes[startB]
  const snapped: [number, number] = [
    startFrom[0] + (startTo[0] - startFrom[0]) * best.fraction,
    startFrom[1] + (startTo[1] - startFrom[1]) * best.fraction,
  ]
  let index = best.index, fraction = best.fraction, remaining = travel
  while (remaining > 0 && index < edges.length) {
    const [a, b] = edges[index]
    const from = network.nodes[a], to = network.nodes[b]
    const length = Math.hypot((to[1] - from[1]) * metersLon, (to[0] - from[0]) * metersLat)
    if (length < 1 || length > 2000) break
    const available = (1 - fraction) * length
    if (remaining <= available) { fraction += remaining / length; remaining = 0; break }
    remaining -= available
    fraction = 1
    if (index + 1 >= edges.length || edges[index + 1][0] !== b) break
    index++
    fraction = 0
  }
  const [a, b] = edges[index]
  const from = network.nodes[a], to = network.nodes[b]
  const blend = Math.max(0, 1 - elapsed / 15)
  return { position: [
    from[0] + (to[0] - from[0]) * fraction + (position[0] - snapped[0]) * blend,
    from[1] + (to[1] - from[1]) * fraction + (position[1] - snapped[1]) * blend,
  ], method: 'route' }
}
