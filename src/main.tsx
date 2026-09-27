import React, { useCallback, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, AlertTriangle, ArrowLeft, BarChart3, BusFront,
  ChevronDown, ChevronRight, Map as MapIcon,
  MapPin, Menu, Route, Search,
  X, PanelRightOpen, PanelRightClose,
} from 'lucide-react'
import TransportMap from './TransportMap'
import plan from './data/plan.json'
import roadRoutes from './data/road_routes.json'
import { describeRoutes, type RouteGroup } from './routes'
import { estimateBetweenFixes, type SharedNetwork } from './timeline'
import { riskStatus, type RiskStatus } from './risk'
import type { DelayExplanationData } from './DelayExplanation'
import WhatIf from './WhatIf'
import './style.css'
import './detail.css'

type Stop = { id: string; time: string; lon: number; lat: number; name: string }
type Vehicle = {
  id: string; position: [number, number]; speed: number;
  heading: number | null; gpsAgeMin: number;
  stops: Stop[]; nextStop: Stop | null; estimateSeconds: number | null;
  forecastTime: string | null; forecastStopId: string | null; sampleId: string | null;
  forecastStop: Stop | null; probability: number | null; stale: boolean;
  forecastStale: boolean;
  forecastExpired: boolean;
  delayExplanation: DelayExplanationData | null;
  reason: string | null; recommendation: string | null; section: string | null;
  source: string; doorStatus: string | null;
  currentDeviationSeconds: number | null;
  meanSpeed5m: number | null; segmentSpeedMeanKmh: number | null;
  stoppedDurationSeconds: number | null;
  forecastAvailability: 'ready' | 'no_point' | 'no_target' | 'bad_gps' | 'off_route' | 'late_packet' | 'stale_gps' | 'ml_unavailable' | 'pending';
  forecastGeneratedAt: string | null; forecastPacketId: string | null;
  nearestForecastPointAt: string | null;
  nearestForecastPointDirection: 'next' | 'previous' | null;
  gpsEventTime: string | null;
  positionMethod: 'gps' | 'route' | 'heading';
}
type View = 'map' | 'vehicles' | 'events' | 'analytics'
type Filter = 'all' | 'normal' | 'deviation' | 'risk'
type Status = RiskStatus

const historicalNetwork = { ...plan.network, roadRoutes } as unknown as SharedNetwork
const historicalRoutes = describeRoutes(plan.vehicles)

function normalizeVehicle(value: Partial<Vehicle> & { id: string; position?: [number, number] | null }): Vehicle | null {
  if (!value.position || value.position.some(point => point === null || !Number.isFinite(point))) return null
  return {
    id: value.id, position: value.position, speed: value.speed ?? 0,
    heading: value.heading ?? null, gpsAgeMin: value.gpsAgeMin ?? 0,
    stops: value.stops ?? [], nextStop: value.nextStop ?? null,
    estimateSeconds: value.estimateSeconds ?? null, forecastTime: value.forecastTime ?? null,
    forecastStopId: value.forecastStopId ?? null, sampleId: value.sampleId ?? null,
    forecastStop: value.forecastStop ?? null, probability: value.probability ?? null,
    delayExplanation: value.delayExplanation ?? null,
    forecastStale: value.forecastStale ?? false,
    forecastExpired: value.forecastExpired ?? false,
    stale: value.stale ?? false, reason: value.reason ?? null,
    recommendation: value.recommendation ?? null, section: value.section ?? null,
    source: value.source ?? 'NDTP', doorStatus: value.doorStatus ?? null,
    currentDeviationSeconds: value.currentDeviationSeconds ?? null,
    meanSpeed5m: value.meanSpeed5m ?? null,
    segmentSpeedMeanKmh: value.segmentSpeedMeanKmh ?? null,
    stoppedDurationSeconds: value.stoppedDurationSeconds ?? null,
    forecastAvailability: value.forecastAvailability ?? 'no_point',
    forecastGeneratedAt: value.forecastGeneratedAt ?? null,
    forecastPacketId: value.forecastPacketId ?? null,
    nearestForecastPointAt: value.nearestForecastPointAt ?? null,
    nearestForecastPointDirection: value.nearestForecastPointDirection ?? null,
    gpsEventTime: value.gpsEventTime ?? null, positionMethod: 'gps',
  }
}

function statusOf(vehicle: Vehicle): Status {
  return riskStatus(vehicle.forecastExpired ? null : vehicle.probability)
}

const statusLabel: Record<Status, string> = {
  normal: 'Низкий риск', minor: 'Средний риск',
  critical: 'Высокий риск', unknown: 'Без прогноза',
}
function minutes(seconds: number | null) {
  if (seconds === null) return '—'
  const absolute = Math.round(Math.abs(seconds))
  const wholeMinutes = Math.floor(absolute / 60)
  const remainder = absolute % 60
  return `${seconds < 0 ? '−' : '+'}${wholeMinutes ? `${wholeMinutes} мин ` : ''}${remainder} с`
}

function stopName(name: string) {
  return name.replace(/,?\s*д\.?\s*\d+.*$/i, '').trim() || name
}

function VehicleRow({ vehicle, active, onClick }: { vehicle: Vehicle; active: boolean; onClick: () => void }) {
  const status = statusOf(vehicle)
  return <button className={`vehicle-row ${active ? 'active' : ''}`} onClick={onClick}>
    <span className={`row-icon ${status}`}><BusFront size={21} strokeWidth={2.1} /></span>
    <span className="row-content">
      <span className="row-heading"><strong>ТС {vehicle.id}</strong><span className={`status-badge ${status}`}>{statusLabelOf(vehicle)}</span></span>
      <span className="row-location"><MapPin size={13} /> {vehicle.nextStop ? stopName(vehicle.nextStop.name) : 'Остановка не указана'}</span>
      <span className="row-meta"><span><Activity size={14} /> {vehicle.speed} км/ч</span><span className={status !== 'unknown' ? `deviation ${status}` : ''}>{vehicle.probability === null ? '—' : `${Math.round(vehicle.probability * 100)}%`}</span></span>
    </span>
    <ChevronRight className="row-chevron" size={18} />
  </button>
}

function statusLabelOf(vehicle: Vehicle): string {
  return vehicle.forecastExpired ? 'Архивный прогноз' : statusLabel[statusOf(vehicle)]
}

function RouteDetail({ id, stops, onBack }: { id: string; stops: Stop[]; onBack: () => void }) {
  return <div className="detail-pane route-detail-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К карте</button>
    <div className="vehicle-card-heading"><div className="vehicle-card-title"><span className="detail-bus"><Route size={24} /></span><div><h2>Маршрут ТС {id}</h2><span className="detail-kicker">Выбранная линия на карте</span></div></div></div>
    <div className="detail-tab-content"><WhatIf key={id} vehicleId={id} stops={stops} /></div>
  </div>
}

function InactiveRouteDetail({ id, onBack }: { id: string; onBack: () => void }) {
  return <div className="detail-pane inactive-route-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К списку ТС</button>
    <div className="detail-heading"><span className="detail-bus"><Route size={27} /></span><div><div className="detail-kicker">Выбранная траектория</div><h2>ТС {id}</h2></div></div>
    <div className="detail-status">Нет актуальной позиции</div>
    <div className="prediction-block"><p>Сейчас от этого ТС нет свежей телеметрии NDTP. На карте показан его маршрут из датасета.</p></div>
  </div>
}

function LiveDetail({ vehicle, onBack }: { vehicle: Vehicle; onBack: () => void }) {
  const knownRoute = vehicle.stops.length > 0
  return <div className="detail-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К списку ТС</button>
    <div className="vehicle-card-heading"><div className="vehicle-card-title"><span className="detail-bus unknown"><BusFront size={25} /></span><div><h2>{knownRoute ? `ТС ${vehicle.id}` : `Устройство ${vehicle.id}`}</h2><span className="detail-kicker">Живой поток NDTP</span></div></div></div>
    <div className="prediction-block"><span className="eyebrow">{knownRoute ? 'Маршрут найден в датасете' : 'Маршрут не найден'}</span><p>{knownRoute ? `Пакет принят и сопоставлен с плановым маршрутом.${vehicle.nextStop ? ` Ближайшая следующая остановка: ${vehicle.nextStop.name}.` : ''} Прогноз задержки по живому потоку пока недоступен.` : 'Идентификатор устройства отсутствует в validate/traffic.csv.'}</p></div>
    <div className="detail-section"><h3>Телеметрия</h3><div className="metric-line"><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div><div className="metric-line"><span>Последний GPS</span><strong>{Math.round(vehicle.gpsAgeMin * 60)} с назад</strong></div><div className="metric-line"><span>Координаты</span><strong>{vehicle.position[0].toFixed(5)}, {vehicle.position[1].toFixed(5)}</strong></div><div className="metric-line"><span>Двери</span><strong>{vehicle.doorStatus ?? 'Нет данных от устройства'}</strong></div><div className="metric-line"><span>Источник</span><strong>{vehicle.source}</strong></div></div>
  </div>
}

function RouteList({ vehicles, routes, select, selectRoute }: {
  vehicles: Vehicle[]; routes: RouteGroup[]; select: (id: string) => void;
  selectRoute: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const byId = new Map(vehicles.map(vehicle => [vehicle.id, vehicle]))
  const listed = new Set(routes.flatMap(route => route.vehicleIds))
  const shown = routes.map(route => ({ ...route, buses: route.vehicleIds.map(id => byId.get(id)).filter((bus): bus is Vehicle => bus !== undefined) }))
    .filter(route => route.buses.length > 0)
  const unassigned = vehicles.filter(vehicle => !listed.has(vehicle.id))
  if (unassigned.length) shown.push({ id: 'unassigned', title: 'Маршрут не указан в данных', vehicleIds: unassigned.map(bus => bus.id), buses: unassigned })
  return <div className="data-view"><div className="data-intro"><h2>Маршруты на линии</h2><p>Линии названы по остановкам расписания: номера маршрутов в выгрузке нет.</p></div>
    <div className="route-list">{shown.length ? shown.map(route => <div className="route-group" key={route.id}>
      <button className={`route-group-button ${expanded === route.id ? 'active' : ''}`} aria-expanded={expanded === route.id} onClick={() => {
        const next = expanded === route.id ? null : route.id
        setExpanded(next)
        selectRoute(route.id === 'unassigned' || !next ? '' : route.buses[0].id)
      }}><span className="route-group-icon"><Route size={19} /></span><span className="route-group-text"><strong>{route.title}</strong><small>{route.buses.length} ТС сейчас</small></span><ChevronDown size={17} /></button>
      {expanded === route.id && <div className="route-buses">{route.buses.map(vehicle => <button key={vehicle.id} className="route-bus-button" onClick={() => select(vehicle.id)}><BusFront size={18} /><span><strong>ТС {vehicle.id}</strong><small>{vehicle.nextStop ? `Следующая: ${stopName(vehicle.nextStop.name)}` : 'Остановка не указана'}</small></span><span className={`status-badge ${statusOf(vehicle)}`}>{statusLabelOf(vehicle)}</span><ChevronRight size={16} /></button>)}</div>}
    </div>) : <div className="empty-state">Сейчас нет транспорта с актуальной телеметрией</div>}</div>
  </div>
}

function DataView({ view, vehicles, select, routes, selectRoute }: {
  view: View; vehicles: Vehicle[]; select: (id: string) => void;
  routes: RouteGroup[]; selectRoute: (id: string) => void;
}) {
  if (view === 'vehicles') return <RouteList vehicles={vehicles} routes={routes} select={select} selectRoute={selectRoute} />
  if (view === 'analytics') {
    const tracked = vehicles.filter(v => v.estimateSeconds !== null)
    return <div className="data-view"><div className="data-intro"><h2>Аналитика движения</h2><p>Вероятность задержки более 2 минут на целевой остановке</p></div><div className="analytics-grid"><div className="analytic-tile"><span>На линии</span><strong>{vehicles.length}</strong><small>ТС с полученной телеметрией</small></div><div className="analytic-tile"><span>С прогнозом</span><strong>{tracked.length}</strong><small>Цель через 10–15 минут</small></div><div className="analytic-tile"><span>Высокий риск</span><strong>{tracked.filter(v => statusOf(v) === 'critical').length}</strong><small>Вероятность от 90%</small></div></div><h3 className="view-subtitle">Транспорт с прогнозом</h3><div className="data-table compact">{tracked.map(v => <button key={v.id} className="analytic-row" onClick={() => select(v.id)}><span className={`tiny-dot ${statusOf(v)}`} /> ТС {v.id}<strong>{Math.round((v.probability ?? 0) * 100)}% · {minutes(v.estimateSeconds)}</strong><ChevronRight size={17} /></button>)}</div></div>
  }
  const title = view === 'events' ? 'События' : 'Транспорт на линии'
  const shown = view === 'events' ? vehicles.filter(v => ['minor', 'critical'].includes(statusOf(v))) : vehicles
  return <div className="data-view"><div className="data-intro"><h2>{title}</h2><p>{view === 'events' ? 'События по текущей телеметрии' : 'Текущие координаты и состояние транспорта'}</p></div><div className="list-grid">{shown.length ? shown.map(v => <button className="grid-row" key={v.id} onClick={() => select(v.id)}><span className={`row-icon ${statusOf(v)}`}><BusFront size={21} /></span><span><strong>ТС {v.id}</strong><small>{v.nextStop ? stopName(v.nextStop.name) : 'Без остановки'}</small></span><span className={`status-badge ${statusOf(v)}`}>{statusLabelOf(v)}</span><ChevronRight size={18} /></button>) : <div className="empty-state">Сейчас событий нет</div>}</div></div>
}

const navItems: { id: View; label: string; icon: React.ElementType }[] = [
  { id: 'map', label: 'Карта', icon: MapIcon },
  { id: 'vehicles', label: 'Транспорт', icon: BusFront },
  { id: 'events', label: 'События', icon: AlertTriangle },
  { id: 'analytics', label: 'Аналитика', icon: BarChart3 },
]

function App() {
  const activePlan = plan
  const network = historicalNetwork
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const [modelStatus, setModelStatus] = useState('ready')
  const [monitorCollapsed, setMonitorCollapsed] = useState(false)
  const [view, setView] = useState<View>('map')
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null)
  const [showRoutes, setShowRoutes] = useState(true)
  const [mobileNav, setMobileNav] = useState(false)
  useEffect(() => {
    let mounted = true
    let busy = false
    const load = async () => {
      if (busy) return
      busy = true
      try {
        const response = await fetch('/api/live/snapshot')
        if (!response.ok) throw new Error(`API ${response.status}`)
        const state = await response.json()
        if (mounted) {
          setVehicles((state.vehicles as (Partial<Vehicle> & { id: string })[]).map(normalizeVehicle).filter((v): v is Vehicle => v !== null))
          setModelStatus(state.modelStatus ?? 'ready')
          setApiError(null)
        }
      } catch (error) {
        if (mounted) { setApiError(error instanceof Error ? error.message : 'Нет связи с API'); setModelStatus('unavailable') }
      } finally { busy = false }
    }
    void load()
    const timer = window.setInterval(() => { void load() }, 1000)
    return () => { mounted = false; window.clearInterval(timer) }
  }, [])
  const displayVehicles = vehicles.map(vehicle => {
    const estimated = estimateBetweenFixes(vehicle.position, vehicle.speed, vehicle.heading,
      Math.max(0, vehicle.gpsAgeMin * 60), network, vehicle.id)
    return { ...vehicle, position: estimated.position, heading: estimated.heading ?? vehicle.heading,
      positionMethod: estimated.method }
  })
  const selected = displayVehicles.find(v => v.id === selectedId) ?? null
  const filtered = displayVehicles.filter(v => {
    const status = statusOf(v)
    if (filter === 'normal' && status !== 'normal') return false
    if (filter === 'deviation' && (v.estimateSeconds === null || Math.abs(v.estimateSeconds) < 60)) return false
    if (filter === 'risk' && !['minor', 'critical'].includes(status)) return false
    return `${v.id} ${v.nextStop?.name ?? ''}`.toLocaleLowerCase('ru').includes(search.toLocaleLowerCase('ru'))
  })
  const countRisk = displayVehicles.filter(v => statusOf(v) === 'critical').length
  const select = useCallback((id: string) => {
    setSelectedId(id || null)
    setSelectedRouteId(null)
    setView('map')
    if (id) setMonitorCollapsed(false)
  }, [])
  const selectRoute = useCallback((id: string) => {
    setSelectedRouteId(id)
    setSelectedId(null)
    setView('map')
    setMonitorCollapsed(false)
  }, [])
  const selectRouteInList = useCallback((id: string) => {
    setSelectedRouteId(id || null)
    setSelectedId(null)
    setMonitorCollapsed(false)
  }, [])
  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'mobile-open' : ''}`}>
      <div className="brand"><strong>МОС<span>ТРАНС</span></strong></div>
      <div className="sidebar-divider" />
      <nav>{navItems.map(item => <button key={item.id} className={`nav-item ${view === item.id ? 'active' : ''}`} onClick={() => { setView(item.id); setMobileNav(false) }}><item.icon size={20} strokeWidth={2} /><span>{item.label}</span>{item.id === 'events' && countRisk > 0 && <em>{countRisk}</em>}</button>)}</nav>
    </aside>
    {mobileNav && <button className="mobile-scrim" aria-label="Закрыть меню" onClick={() => setMobileNav(false)} />}
    <main className="main-area">
      <header className="topbar"><div className="topbar-title"><button className="mobile-menu" aria-label="Открыть меню" onClick={() => setMobileNav(true)}><Menu size={22} /></button><h1>{view === 'map' ? 'Карта' : navItems.find(item => item.id === view)?.label}</h1></div></header>
      {apiError && <div className="service-banner">Нет связи с backend: {apiError}. Показано последнее полученное состояние.</div>}
      {modelStatus === 'unavailable' && !apiError && <div className="service-banner">ML-сервис недоступен: положение ТС обновляется, новые прогнозы временно не рассчитываются.</div>}
      <div className="workspace">
        <section className="primary-panel">
          {view === 'map' ? <TransportMap vehicles={displayVehicles} selected={selected} selectedId={selectedRouteId ?? selectedId} onSelect={select} onRouteSelect={selectRoute} showRoutes={showRoutes} setShowRoutes={setShowRoutes} collapsed={monitorCollapsed} network={network} replayTime={null} /> : <DataView view={view} vehicles={displayVehicles} select={select} routes={historicalRoutes} selectRoute={selectRouteInList} />}
        </section>
        {monitorCollapsed && <button className="restore-monitor" onClick={() => setMonitorCollapsed(false)} title="Открыть мониторинг" aria-label="Открыть мониторинг"><PanelRightOpen size={19} /></button>}
        {monitorCollapsed ? null : <aside className="right-panel">
          {selectedRouteId ? <>
            <button className="collapse-detail" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг" aria-label="Свернуть мониторинг"><PanelRightClose size={18} /></button>
            <RouteDetail id={selectedRouteId} stops={activePlan.vehicles.find(vehicle => vehicle.id === selectedRouteId)?.stops ?? []} onBack={() => setSelectedRouteId(null)} />
          </> : selectedId ? <>
            <button className="collapse-detail" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг" aria-label="Свернуть мониторинг"><PanelRightClose size={18} /></button>
            {selected ? <LiveDetail vehicle={selected} onBack={() => setSelectedId(null)} /> : <InactiveRouteDetail id={selectedId} onBack={() => setSelectedId(null)} />}
          </> : <>
            <div className="panel-header">
              <button className="monitor-heading" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг"><span className="eyebrow">МОНИТОРИНГ</span><span className="monitor-title">Транспорт на линии <span>{displayVehicles.length}</span></span></button>
              <button className="panel-icon-button" title="Свернуть мониторинг" aria-label="Свернуть мониторинг" onClick={() => setMonitorCollapsed(true)}><PanelRightClose size={19} /></button>
            </div>
            <div className="search-box"><Search size={18} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Номер ТС или остановка" aria-label="Поиск транспорта" />{search && <button aria-label="Очистить поиск" onClick={() => setSearch('')}><X size={16} /></button>}</div>
            <div className="filter-row"><button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>Все</button><button className={filter === 'deviation' ? 'active' : ''} onClick={() => setFilter('deviation')}>Отклонения</button><button className={filter === 'risk' ? 'active' : ''} onClick={() => setFilter('risk')}>Риск</button></div>
            <div className="vehicle-list">{filtered.length ? filtered.map(vehicle => <VehicleRow key={vehicle.id} vehicle={vehicle} active={false} onClick={() => setSelectedId(vehicle.id)} />) : <div className="empty-state">Транспорт не найден</div>}</div>
            <div className="panel-footer"><span className="footer-dot" /> {filtered.length} из {displayVehicles.length} ТС · прямой эфир</div>
          </>}
        </aside>}
      </div>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
