type ScheduledStop = { time: string; name: string }

export function scheduledCycleMinutes(stops: ScheduledStop[]): number | null {
  if (stops.length < 2) return null
  const anchor = stops[0].name
  const visits: number[] = []
  for (const stop of stops) {
    if (stop.name !== anchor) continue
    const at = Date.parse(stop.time.replace(' ', 'T')) / 60000
    if (Number.isFinite(at) && (!visits.length || at - visits[visits.length - 1] >= 20)) visits.push(at)
  }
  const cycles = visits.slice(1).map((at, index) => at - visits[index])
    .filter(minutes => minutes >= 20 && minutes <= 240).sort((a, b) => a - b)
  if (!cycles.length) return null
  return Math.round(cycles[Math.floor(cycles.length / 2)])
}

export function serviceScenario(cycleMinutes: number, currentBuses: number, addedBuses: number) {
  const beforeHeadway = cycleMinutes / currentBuses
  const afterHeadway = cycleMinutes / (currentBuses + addedBuses)
  return {
    beforeHeadway, afterHeadway,
    beforeWait: beforeHeadway / 2, afterWait: afterHeadway / 2,
    beforeDepartures: 60 / beforeHeadway, afterDepartures: 60 / afterHeadway,
  }
}
