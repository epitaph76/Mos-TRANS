import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(new URL('../src/timeline.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } })
const { positionAt, formatTime, stopSeconds, forecastAt, indexNetwork, snapToNetwork, snapToVehicleRoute,
  estimateBetweenFixes, matchRoadFix, pointOnRoadRoute } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
const riskSource = readFileSync(new URL('../src/risk.ts', import.meta.url), 'utf8')
const riskModule = ts.transpileModule(riskSource, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } })
const { riskStatus } = await import(`data:text/javascript;base64,${Buffer.from(riskModule.outputText).toString('base64')}`)
const scenarioSource = readFileSync(new URL('../src/whatIfModel.ts', import.meta.url), 'utf8')
const scenarioModule = ts.transpileModule(scenarioSource, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } })
const { scheduledCycleMinutes, serviceScenario } = await import(`data:text/javascript;base64,${Buffer.from(scenarioModule.outputText).toString('base64')}`)
const routesSource = readFileSync(new URL('../src/routes.ts', import.meta.url), 'utf8')
const routesModule = ts.transpileModule(routesSource, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } })
const { describeRoutes } = await import(`data:text/javascript;base64,${Buffer.from(routesModule.outputText).toString('base64')}`)
const track = [[10, 55, 37, 20, 0], [30, 55.002, 37.004, 25, 90], [600, 56, 38, 0, null]]

test('risk colors change only at 70% and 90%', () => {
  assert.equal(riskStatus(0.699), 'normal')
  assert.equal(riskStatus(0.7), 'minor')
  assert.equal(riskStatus(0.899), 'minor')
  assert.equal(riskStatus(0.9), 'critical')
  assert.equal(riskStatus(null), 'unknown')
})

test('what-if uses scheduled returns and the specified fleet', () => {
  const stops = ['02:45', '03:54', '04:00', '05:11', '05:15', '06:32']
    .map(time => ({ name: 'Начальная остановка', time: `2026-01-06 ${time}:00` }))
  assert.equal(scheduledCycleMinutes(stops), 77)
  const scenario = serviceScenario(80, 4, 1)
  assert.equal(scenario.beforeHeadway, 20)
  assert.equal(scenario.afterHeadway, 16)
  assert.equal(scenario.beforeWait, 10)
  assert.equal(scenario.afterWait, 8)
})

test('transport tab groups vehicles sharing scheduled endpoints', () => {
  const outbound = [{ name: 'Начальная ул., д.1', lat: 55, lon: 37 },
    { name: 'Конечная ул., д.2', lat: 55.01, lon: 37.02 }]
  const routes = describeRoutes([
    { id: 'bus-a', stops: outbound },
    { id: 'bus-b', stops: [...outbound].reverse() },
  ])
  assert.equal(routes.length, 1)
  assert.deepEqual(routes[0].vehicleIds, ['bus-a', 'bus-b'])
})

test('positions use only observed fixes, never the next GPS packet', () => {
  assert.equal(positionAt(track, 9), null)
  assert.deepEqual(positionAt(track, 10).position, [55, 37])
  assert.deepEqual(positionAt(track, 20).position, [55, 37])
  assert.deepEqual(positionAt(track, 30).position, [55.002, 37.004])
})

test('long telemetry gaps hold the last fix then hide the vehicle', () => {
  assert.deepEqual(positionAt(track, 100).position, [55.002, 37.004])
  assert.equal(positionAt(track, 211), null)
  assert.deepEqual(positionAt(track, 600).position, [56, 38])
  assert.equal(positionAt(track, 781), null)
})

test('midnight schedule entries retain the next-day offset', () => {
  assert.equal(stopSeconds('2026-01-07 00:01:00', '2026-01-06'), 86460)
  assert.equal(formatTime(2, true), '00:00:02')
  assert.equal(formatTime(86398, true), '23:59:58')
})

test('forecast uses no future point and stays within a 10–15 minute horizon', () => {
  const points = [{ time: 42000, forecastTime: '2026-01-06 11:52:00', estimateSeconds: 60 }]
  assert.equal(forecastAt(points, 41999, '2026-01-06'), undefined)
  assert.equal(forecastAt(points, 42000, '2026-01-06'), points[0])
  assert.equal(forecastAt(points, 42059, '2026-01-06'), points[0])
  assert.equal(forecastAt(points, 42060, '2026-01-06'), undefined)
  assert.equal(forecastAt([{ ...points[0], forecastTime: '2026-01-06 11:50:00' }], 42001, '2026-01-06'), undefined)
})

test('browser asset contains planned stops and no future GPS track', () => {
  const data = JSON.parse(readFileSync(new URL('../src/data/plan.json', import.meta.url), 'utf8'))
  const roads = JSON.parse(readFileSync(new URL('../src/data/road_routes.json', import.meta.url), 'utf8'))
  assert(data.start > 0)
  assert(data.end > data.start)
  for (const vehicle of data.vehicles) {
    assert(!('track' in vehicle))
    assert(vehicle.stops.every((stop, index) => index === 0 || stop.time >= vehicle.stops[index - 1].time))
  }
  assert(data.network.nodes.length > 0)
  assert(data.network.edges.length > 0)
  assert(data.network.edges.every(([a, b, owners]) => a !== b && owners.length > 0))
  assert(data.vehicles.every(vehicle => roads.vehicles[vehicle.id]?.length === vehicle.stops.length - 1))
  assert(roads.segments.every(segment => segment.length >= 2))
  const pairs = Object.values(roads.vehicles).flat()
  assert(pairs.filter(index => index >= 0).length / pairs.length > 0.95)
})

test('GPS position snaps only to a nearby corridor owned by the vehicle', () => {
  const network = indexNetwork({ mergeMeters: 35,
    nodes: [[55, 37], [55, 37.001]], edges: [[0, 1, ['bus-a']]] })
  const point = [55.0001, 37.0005]
  assert(Math.abs(snapToNetwork(point, 90, network, 'bus-a')[0] - 55) < 1e-6)
  assert.deepEqual(snapToNetwork(point, 90, network, 'bus-b'), point)
  assert.deepEqual(snapToNetwork([55.01, 37.01], 90, network, 'bus-a'), [55.01, 37.01])
})

test('marker moves from the last received fix along the planned route, then accepts the next fix', () => {
  const network = { mergeMeters: 0, nodes: [[55, 37], [55, 37.002], [55.001, 37.002]],
    edges: [[0, 1, ['bus']], [1, 2, ['bus']]] }
  const start = [55, 37]
  const after5 = estimateBetweenFixes(start, 36, 90, 5, network, 'bus')
  const after20 = estimateBetweenFixes(start, 36, 90, 20, network, 'bus')
  assert.equal(after5.method, 'route')
  assert(after5.position[1] > start[1])
  assert(after20.position[0] > start[0])
  assert(Math.abs(estimateBetweenFixes([55.0002, 37], 18, 90, 15, network, 'bus').position[0] - 55) < 1e-8)
  assert.deepEqual(estimateBetweenFixes([55.0002, 37.0021], 20, 0, 0, network, 'bus').position,
    [55.0002, 37.0021])
  assert.deepEqual(estimateBetweenFixes(start, 36, 90, -1, network, 'bus').position, start)
  assert.deepEqual(estimateBetweenFixes(start, 0, 90, 10, network, 'bus').position, start)
  assert.deepEqual(estimateBetweenFixes(start, 36, 90, 40, network, 'bus').position,
    estimateBetweenFixes(start, 36, 90, 30, network, 'bus').position)
})

test('road geometry is used for both GPS snapping and travel around a bend', () => {
  const network = { mergeMeters: 0, nodes: [], edges: [], roadRoutes: {
    segments: [[[55, 37], [55, 37.001], [55.001, 37.001]]], vehicles: { bus: [0] },
  } }
  const gps = [55.0001, 37.0001]
  const snapped = estimateBetweenFixes(gps, 36, 90, 0, network, 'bus')
  assert.equal(snapped.method, 'gps')
  assert(Math.abs(snapped.position[0] - 55) < 1e-8)
  const moving = estimateBetweenFixes(gps, 36, 90, 12, network, 'bus')
  assert.equal(moving.method, 'route')
  assert(Math.abs(moving.position[1] - 37.001) < 1e-8)
  assert(moving.position[0] > 55)
  const distant = estimateBetweenFixes([55.01, 37.01], 36, 90, 5, network, 'bus')
  assert.equal(distant.method, 'route')
  assert.deepEqual(snapToVehicleRoute(distant.position, network, 'bus'), distant.position)
  assert.deepEqual(snapToVehicleRoute([55.0005, 37.0005], network, 'bus'), [55.0005, 37.001])
})

test('live е66 stays on its direction and does not jump across nearby road legs', () => {
  const recording = JSON.parse(readFileSync(new URL('../mos_trans/replay_data/e66.json', import.meta.url), 'utf8'))
  const roadRoutes = recording.roadRoutes
  const network = { mergeMeters: 35, nodes: [], edges: [], roadRoutes: {
    segments: roadRoutes.segments, vehicles: {
      A: roadRoutes.directions.A, B: roadRoutes.directions.B,
    },
  } }
  const meters = (a, b) => Math.hypot((a[0] - b[0]) * 111_320,
    (a[1] - b[1]) * 111_320 * Math.cos(a[0] * Math.PI / 180))
  for (const trip of recording.trips) {
    const start = Date.parse(trip.stops[0].time)
    const end = Date.parse(trip.stops.at(-1).time)
    const fixes = recording.points.filter(point => {
      const at = Date.parse(point.time)
      return start <= at && at <= end
    })
    let previous = null
    for (const fix of fixes) {
      const raw = [fix.lat, fix.lon]
      const dt = previous ? (Date.parse(fix.time) - Date.parse(previous.fix.time)) / 1000 : Infinity
      const rawMove = previous ? meters(raw, previous.raw) : 0
      const continuous = previous && dt <= 180
      const match = matchRoadFix(raw, fix.heading, fix.speed ?? 0, network, trip.direction,
        continuous ? previous.match.progressMeters : undefined,
        Math.max(60, rawMove * 2 + 40), continuous ? previous.match.position : undefined)
      assert(match, `${trip.id} has a road match at ${fix.time}`)
      assert(match.distanceMeters < 120)
      assert(meters(pointOnRoadRoute(network, trip.direction, match.progressMeters), match.position) < 3)
      if (continuous && dt <= 45)
        assert(meters(match.position, previous.match.position) <= rawMove + 45,
          `${trip.id} jumped at ${fix.time}`)
      previous = { fix, raw, match }
    }
  }
})
