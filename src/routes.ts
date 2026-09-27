type Stop = { name: string; lat: number; lon: number }
type PlannedVehicle = { id: string; stops: Stop[] }

export type RouteGroup = { id: string; title: string; vehicleIds: string[] }

function shortName(name: string): string {
  return name.replace(/,?\s*д\.?\s*\d+.*$/i, '').trim() || name
}

function routeEndpoints(vehicle: PlannedVehicle): [string, string] | null {
  const named = vehicle.stops.filter(stop => stop.name !== 'Остановка')
  if (named.length < 2) return null
  const origin = named[0]
  const destination = named.reduce((farthest, stop) => {
    const distance = (stop.lat - origin.lat) ** 2 + ((stop.lon - origin.lon) * 0.56) ** 2
    return distance > farthest.distance ? { stop, distance } : farthest
  }, { stop: origin, distance: 0 }).stop
  const first = shortName(origin.name)
  const last = shortName(destination.name)
  return first === last ? null : [first, last]
}

/** The export has no public route numbers, so lines are named by scheduled endpoints. */
export function describeRoutes(vehicles: PlannedVehicle[]): RouteGroup[] {
  const routes = new Map<string, RouteGroup>()
  for (const vehicle of vehicles) {
    const endpoints = routeEndpoints(vehicle)
    const key = endpoints ? [...endpoints].sort((a, b) => a.localeCompare(b, 'ru')).join('|') : `vehicle:${vehicle.id}`
    let route = routes.get(key)
    if (!route) {
      route = { id: vehicle.id, title: endpoints ? `${endpoints[0]} — ${endpoints[1]}` : `Линия ТС ${vehicle.id}`, vehicleIds: [] }
      routes.set(key, route)
    }
    route.vehicleIds.push(vehicle.id)
  }
  return [...routes.values()].sort((a, b) => a.title.localeCompare(b.title, 'ru'))
}
