import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(new URL('../src/timeline.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } })
const { positionAt, formatTime, stopSeconds, forecastAt, indexNetwork, snapToNetwork, estimateBetweenFixes } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
const track = [[10, 55, 37, 20, 0], [30, 55.002, 37.004, 25, 90], [600, 56, 38, 0, null]]

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
  assert(data.start > 0)
  assert(data.end > data.start)
  for (const vehicle of data.vehicles) {
    assert(!('track' in vehicle))
    assert(vehicle.stops.every((stop, index) => index === 0 || stop.time >= vehicle.stops[index - 1].time))
  }
  assert(data.network.nodes.length > 0)
  assert(data.network.edges.length > 0)
  assert(data.network.edges.every(([a, b, owners]) => a !== b && owners.length > 0))
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
