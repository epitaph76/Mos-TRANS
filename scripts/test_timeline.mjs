import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(new URL('../src/timeline.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } })
const { positionAt, formatTime, stopSeconds, forecastAt, indexNetwork, snapToNetwork } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
const track = [[10, 55, 37, 20, 0], [30, 55.002, 37.004, 25, 90], [600, 56, 38, 0, null]]

test('positions are bounded by observed fixes and interpolate short gaps', () => {
  assert.equal(positionAt(track, 9), null)
  assert.deepEqual(positionAt(track, 10).position, [55, 37])
  const midpoint = positionAt(track, 20).position
  assert(Math.abs(midpoint[0] - 55.001) < 1e-10)
  assert(Math.abs(midpoint[1] - 37.002) < 1e-10)
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

test('generated timeline spans every cleaned real track with strictly ordered fixes', () => {
  const data = JSON.parse(readFileSync(new URL('../src/data/timeline.json', import.meta.url), 'utf8'))
  assert.equal(data.start, Math.min(...data.vehicles.map(vehicle => vehicle.track[0][0])))
  assert.equal(data.end, Math.max(...data.vehicles.map(vehicle => vehicle.track.at(-1)[0])))
  assert.equal(data.start, 2)
  for (const vehicle of data.vehicles) {
    assert(vehicle.track.every((point, index) => index === 0 || point[0] > vehicle.track[index - 1][0]))
    assert(positionAt(vehicle.track, vehicle.track[0][0]))
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
