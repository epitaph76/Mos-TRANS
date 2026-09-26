import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, AlertTriangle, ArrowLeft, BarChart3, BusFront, CalendarDays,
  ChevronDown, ChevronRight, Clock3, Map as MapIcon,
  MapPin, Menu, Route, Search,
  X, Play, Pause, PanelRightOpen, PanelRightClose,
} from 'lucide-react'
import TransportMap from './TransportMap'
import timeline from './data/timeline.json'
import { forecastAt, formatTime, indexNetwork, positionAt, snapToNetwork, stopSeconds, type SharedNetwork, type TimelineVehicle } from './timeline'
import './style.css'
import './detail.css'

type Stop = { id: string; time: string; lon: number; lat: number; name: string }
type Vehicle = {
  id: string; position: [number, number]; speed: number;
  heading: number | null; gpsAgeMin: number;
  stops: Stop[]; nextStop: Stop | null; estimateSeconds: number | null;
  forecastTime: string | null; forecastStopId: string | null; sampleId: string | null;
}
type View = 'map' | 'schedule' | 'routes' | 'vehicles' | 'events' | 'analytics'
type Filter = 'all' | 'normal' | 'deviation' | 'risk'
type Status = 'normal' | 'minor' | 'delay' | 'critical' | 'early' | 'unknown'

const tracks = timeline.vehicles as TimelineVehicle[]
const network = timeline.network as SharedNetwork
const networkIndex = indexNetwork(network)
const startTime = timeline.start
const endTime = timeline.end
const playbackSpeeds = [1, 15, 60, 300, 900]

function vehiclesAt(time: number): Vehicle[] {
  const vehicles: Vehicle[] = []
  for (const item of tracks) {
    const location = positionAt(item.track, time)
    if (!location) continue
    const nextStop = item.stops.find(stop => stopSeconds(stop.time, timeline.date) >= time) ?? null
    const forecast = forecastAt(item.forecasts, time, timeline.date)
    vehicles.push({
      id: item.id, position: snapToNetwork(location.position, location.heading, networkIndex, item.id),
      speed: location.point[3],
      heading: location.heading, gpsAgeMin: location.age / 60,
      stops: item.stops.filter(stop => Math.abs(stopSeconds(stop.time, timeline.date) - time) <= 45 * 60).slice(0, 10),
      nextStop, estimateSeconds: forecast?.estimateSeconds ?? null,
      forecastTime: forecast?.forecastTime ?? null,
      forecastStopId: forecast?.forecastStopId ?? null,
      sampleId: forecast?.sampleId ?? null,
    })
  }
  return vehicles.sort((a, b) => (a.estimateSeconds === null ? 1 : 0) - (b.estimateSeconds === null ? 1 : 0) || a.id.localeCompare(b.id))
}

function statusOf(vehicle: Vehicle): Status {
  const seconds = vehicle.estimateSeconds
  if (seconds === null) return 'unknown'
  if (seconds < -120) return 'early'
  if (seconds <= 60) return 'normal'
  if (seconds <= 120) return 'minor'
  if (seconds <= 300) return 'delay'
  return 'critical'
}

const statusLabel: Record<Status, string> = {
  normal: 'По графику', minor: 'Небольшое отклонение', delay: 'Задержка',
  critical: 'Критическая задержка', early: 'Опережение', unknown: 'Без оценки',
}
function minutes(seconds: number | null) {
  if (seconds === null) return '—'
  const value = Math.round(Math.abs(seconds) / 60)
  return `${seconds < 0 ? '−' : '+'}${value} мин`
}

function stopName(name: string) {
  return name.replace(/,?\s*д\.?\s*\d+.*$/i, '').trim() || name
}

function timeOnly(value: string | null) {
  return value ? value.slice(11, 16) : '—'
}

function VehicleRow({ vehicle, active, onClick }: { vehicle: Vehicle; active: boolean; onClick: () => void }) {
  const status = statusOf(vehicle)
  return <button className={`vehicle-row ${active ? 'active' : ''}`} onClick={onClick}>
    <span className={`row-icon ${status}`}><BusFront size={21} strokeWidth={2.1} /></span>
    <span className="row-content">
      <span className="row-heading"><strong>ТС {vehicle.id}</strong><span className={`status-badge ${status}`}>{statusLabel[status]}</span></span>
      <span className="row-location"><MapPin size={13} /> {vehicle.nextStop ? stopName(vehicle.nextStop.name) : 'Остановка не указана'}</span>
      <span className="row-meta"><span><Clock3 size={14} /> {vehicle.nextStop ? timeOnly(vehicle.nextStop.time) : '—'}</span><span><Activity size={14} /> {vehicle.speed} км/ч</span><span className={status !== 'unknown' ? `deviation ${status}` : ''}>{minutes(vehicle.estimateSeconds)}</span></span>
    </span>
    <ChevronRight className="row-chevron" size={18} />
  </button>
}

function VehicleDetail({ vehicle, time, onBack }: { vehicle: Vehicle; time: number; onBack: () => void }) {
  const status = statusOf(vehicle)
  const [tab, setTab] = useState<'tracking' | 'analytics' | 'details'>('tracking')
  const windowStart = time - 45 * 60
  const windowEnd = time + 45 * 60
  const stopEvents = vehicle.stops
    .map(stop => ({ stop, timestamp: stopSeconds(stop.time, timeline.date) }))
    .filter(event => event.timestamp >= windowStart && event.timestamp <= windowEnd)
  return <div className="detail-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К списку ТС</button>
    <div className="vehicle-card-heading">
      <div className="vehicle-card-title"><span className={`detail-bus ${status}`}><BusFront size={25} /></span><div><h2>ТС {vehicle.id}</h2><span className="detail-kicker">Московский наземный транспорт</span></div></div>
      <span className={`detail-status ${status}`}><span className="status-dot" />{statusLabel[status]}</span>
    </div>
    <div className="vehicle-meta-grid">
      <div><span>Ближайшая остановка</span><strong>{vehicle.nextStop ? stopName(vehicle.nextStop.name) : 'Нет данных'}</strong></div>
      <div><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div>
      <div><span>Последний GPS</span><strong>{vehicle.gpsAgeMin < 1 ? 'менее минуты' : `${Math.round(vehicle.gpsAgeMin)} мин`}</strong></div>
    </div>
    <div className="detail-tabs" role="tablist" aria-label="Информация о транспортном средстве">
      <button role="tab" aria-selected={tab === 'tracking'} className={tab === 'tracking' ? 'active' : ''} onClick={() => setTab('tracking')}>Маршрут</button>
      <button role="tab" aria-selected={tab === 'analytics'} className={tab === 'analytics' ? 'active' : ''} onClick={() => setTab('analytics')}>Аналитика</button>
      <button role="tab" aria-selected={tab === 'details'} className={tab === 'details' ? 'active' : ''} onClick={() => setTab('details')}>Данные</button>
    </div>
    {tab === 'tracking' && <div className="detail-tab-content" role="tabpanel">
      <div className="tracking-heading"><strong>Остановки по расписанию</strong><span>{formatTime(time)} МСК</span></div>
      <div className="tracking-ruler" aria-label="Интервал от 45 минут до и после выбранного времени">
        <span>−45 мин</span><span>−15 мин</span><span className="tracking-now">Сейчас</span><span>+15 мин</span><span>+45 мин</span>
        <div className="tracking-track"><i className="tracking-current" />{stopEvents.map(({ stop, timestamp }, index) => <i key={`${stop.id}-${index}`} className="tracking-stop" style={{ left: `${(timestamp - windowStart) / (windowEnd - windowStart) * 100}%` }} title={`${timeOnly(stop.time)} · ${stopName(stop.name)}`} />)}</div>
      </div>
      {vehicle.nextStop && <div className="next-stop-feature"><MapPin size={18} /><div><span>Следующая остановка</span><strong>{stopName(vehicle.nextStop.name)}</strong><small>По расписанию · {timeOnly(vehicle.nextStop.time)}</small></div></div>}
      <div className="stop-event-list">{stopEvents.length ? stopEvents.slice(0, 6).map(({ stop, timestamp }, index) => <div className="stop-event" key={`${stop.id}-${index}`}><span className={`stop-event-dot ${timestamp <= time ? 'past' : ''}`} /><time>{timeOnly(stop.time)}</time><span>{stopName(stop.name)}</span><small>{timestamp <= time ? 'По расписанию' : 'Далее по графику'}</small></div>) : <p className="empty-stop-events">Нет остановок в интервале ±45 минут от выбранного времени.</p>}</div>
    </div>}
    {tab === 'analytics' && <div className="detail-tab-content" role="tabpanel">
      <div className="prediction-block"><span className="eyebrow">Оценка на 10–15 минут</span><strong className={status}>{minutes(vehicle.estimateSeconds)}</strong><p>{vehicle.estimateSeconds === null ? 'Для этого ТС в выбранный момент нет прогнозной точки.' : 'Пока отображается текущее отклонение из исторических данных; ML-модель не подключена.'}</p></div>
      <div className="detail-section"><h3>Показатели движения</h3><div className="metric-line"><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div><div className="metric-line"><span>Отклонение от графика</span><strong>{minutes(vehicle.estimateSeconds)}</strong></div></div>
    </div>}
    {tab === 'details' && <div className="detail-tab-content" role="tabpanel">
      <div className="detail-section"><h3>Телеметрия</h3><div className="metric-line"><span>Последний GPS</span><strong>{vehicle.gpsAgeMin < 1 ? 'менее минуты назад' : `${Math.round(vehicle.gpsAgeMin)} мин назад`}</strong></div><div className="metric-line"><span>Координаты</span><strong>GPS-трек NDTP</strong></div></div>
      {vehicle.forecastTime && <div className="detail-section"><h3>Прогнозная точка</h3><div className="metric-line"><span>Целевая остановка</span><strong>{timeOnly(vehicle.forecastTime)}</strong></div><div className="metric-line"><span>ID точки</span><strong>{vehicle.sampleId}</strong></div></div>}
      <div className="detail-section"><h3>Данные</h3><div className="metric-line"><span>Источник</span><strong>Исторический датасет NDTP</strong></div><div className="metric-line"><span>Время среза</span><strong>{formatTime(time)} МСК</strong></div></div>
    </div>}
  </div>
}

function InactiveRouteDetail({ id, onBack }: { id: string; onBack: () => void }) {
  return <div className="detail-pane inactive-route-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К списку ТС</button>
    <div className="detail-heading"><span className="detail-bus"><Route size={27} /></span><div><div className="detail-kicker">Выбранная траектория</div><h2>ТС {id}</h2></div></div>
    <div className="detail-status">Нет актуальной позиции</div>
    <div className="prediction-block"><p>В выбранный момент свежей телеметрии нет. На карте показана траектория по доступным GPS-точкам за день. Выберите другое время на шкале, чтобы увидеть движение ТС.</p></div>
  </div>
}

function DataView({ view, vehicles, select }: { view: View; vehicles: Vehicle[]; select: (id: string) => void }) {
  if (view === 'schedule') {
    const rows = vehicles.filter(v => v.nextStop).sort((a, b) => (a.nextStop?.time || '').localeCompare(b.nextStop?.time || ''))
    return <div className="data-view"><div className="data-intro"><h2>Ближайшие прибытия</h2><p>Плановое расписание для транспорта на линии</p></div><div className="data-table"><div className="table-header"><span>Время</span><span>Транспорт</span><span>Остановка</span><span>Состояние</span></div>{rows.map(v => <button key={v.id} className="table-row" onClick={() => select(v.id)}><strong>{timeOnly(v.nextStop!.time)}</strong><span>ТС {v.id}</span><span>{stopName(v.nextStop!.name)}</span><span className={`status-badge ${statusOf(v)}`}>{statusLabel[statusOf(v)]}</span></button>)}</div></div>
  }
  if (view === 'analytics') {
    const tracked = vehicles.filter(v => v.estimateSeconds !== null)
    return <div className="data-view"><div className="data-intro"><h2>Аналитика движения</h2><p>Срез по выбранному времени · оценка основана на текущем отклонении</p></div><div className="analytics-grid"><div className="analytic-tile"><span>На линии</span><strong>{vehicles.length}</strong><small>ТС с актуальной телеметрией</small></div><div className="analytic-tile"><span>С оценкой</span><strong>{tracked.length}</strong><small>Прогнозные точки в окне 15 минут</small></div><div className="analytic-tile"><span>Требуют внимания</span><strong>{tracked.filter(v => ['delay', 'critical'].includes(statusOf(v))).length}</strong><small>Положительное отклонение от графика</small></div></div><h3 className="view-subtitle">Транспорт с оценкой</h3><div className="data-table compact">{tracked.map(v => <button key={v.id} className="analytic-row" onClick={() => select(v.id)}><span className={`tiny-dot ${statusOf(v)}`} /> ТС {v.id}<strong>{minutes(v.estimateSeconds)}</strong><ChevronRight size={17} /></button>)}</div></div>
  }
  const title = view === 'routes' ? 'Маршрутная сеть' : view === 'events' ? 'События' : 'Транспорт на линии'
  const shown = view === 'events' ? vehicles.filter(v => ['minor', 'delay', 'critical', 'early'].includes(statusOf(v))) : vehicles
  return <div className="data-view"><div className="data-intro"><h2>{title}</h2><p>{view === 'routes' ? 'Траектории восстановлены по GPS-телеметрии' : view === 'events' ? 'Отклонения от расписания в выбранном срезе' : 'Текущие координаты и состояние транспорта'}</p></div><div className="list-grid">{shown.length ? shown.map(v => <button className="grid-row" key={v.id} onClick={() => select(v.id)}><span className={`row-icon ${statusOf(v)}`}><BusFront size={21} /></span><span><strong>ТС {v.id}</strong><small>{v.nextStop ? stopName(v.nextStop.name) : 'Без остановки'}</small></span><span className={`status-badge ${statusOf(v)}`}>{statusLabel[statusOf(v)]}</span><ChevronRight size={18} /></button>) : <div className="empty-state">В этом срезе событий нет</div>}</div></div>
}

const navItems: { id: View; label: string; icon: React.ElementType }[] = [
  { id: 'map', label: 'Карта', icon: MapIcon },
  { id: 'schedule', label: 'Расписание', icon: CalendarDays },
  { id: 'routes', label: 'Маршруты', icon: Route },
  { id: 'vehicles', label: 'Транспорт', icon: BusFront },
  { id: 'events', label: 'События', icon: AlertTriangle },
  { id: 'analytics', label: 'Аналитика', icon: BarChart3 },
]

function App() {
  const [time, setTime] = useState(startTime)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(60)
  const [monitorCollapsed, setMonitorCollapsed] = useState(false)
  const [view, setView] = useState<View>('map')
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [showRoutes, setShowRoutes] = useState(true)
  const [mobileNav, setMobileNav] = useState(false)
  useEffect(() => {
    if (!playing) return
    let previous = performance.now()
    const timer = window.setInterval(() => {
      const now = performance.now()
      const elapsed = Math.min((now - previous) / 1000, 1)
      previous = now
      setTime(current => Math.min(endTime, current + elapsed * speed))
    }, 100)
    return () => window.clearInterval(timer)
  }, [playing, speed])
  useEffect(() => {
    if (time >= endTime) setPlaying(false)
  }, [time])
  const vehicles = useMemo(() => vehiclesAt(time), [time])
  const selected = vehicles.find(v => v.id === selectedId) ?? null
  const filtered = vehicles.filter(v => {
    const status = statusOf(v)
    if (filter === 'normal' && status !== 'normal') return false
    if (filter === 'deviation' && !['minor', 'delay', 'early'].includes(status)) return false
    if (filter === 'risk' && !['delay', 'critical'].includes(status)) return false
    return `${v.id} ${v.nextStop?.name ?? ''}`.toLocaleLowerCase('ru').includes(search.toLocaleLowerCase('ru'))
  })
  const countRisk = vehicles.filter(v => ['delay', 'critical'].includes(statusOf(v))).length
  const select = useCallback((id: string) => {
    setSelectedId(id || null)
    setView('map')
    if (id) setMonitorCollapsed(false)
  }, [])
  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'mobile-open' : ''}`}>
      <div className="brand"><strong>МОС<span>ТРАНС</span></strong></div>
      <div className="sidebar-divider" />
      <nav>{navItems.map(item => <button key={item.id} className={`nav-item ${view === item.id ? 'active' : ''}`} onClick={() => { setView(item.id); setMobileNav(false) }}><item.icon size={20} strokeWidth={2} /><span>{item.label}</span>{item.id === 'events' && countRisk > 0 && <em>{countRisk}</em>}</button>)}</nav>
    </aside>
    {mobileNav && <button className="mobile-scrim" aria-label="Закрыть меню" onClick={() => setMobileNav(false)} />}
    <main className="main-area">
      <header className="topbar"><div className="topbar-title"><button className="mobile-menu" aria-label="Открыть меню" onClick={() => setMobileNav(true)}><Menu size={22} /></button><h1>{view === 'map' ? 'Карта' : navItems.find(item => item.id === view)?.label}</h1></div><div className="topbar-actions"><div className="date-chip"><CalendarDays size={16} /> 06 января 2026</div><div className="time-select"><Clock3 size={17} />{formatTime(time, true)} МСК</div></div></header>
      <div className="timeline-bar"><button className="play-button" onClick={() => { if (time >= endTime) setTime(startTime); setPlaying(!playing || time >= endTime) }} aria-label={playing ? 'Пауза' : 'Воспроизвести'} title={playing ? 'Пауза' : 'Воспроизвести'}>{playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}</button><span className="timeline-time">{formatTime(time, true)}</span><input className="time-slider" type="range" min={startTime} max={endTime} step="1" value={time} onChange={event => { setPlaying(false); setTime(Number(event.target.value)) }} aria-valuetext={`${formatTime(time, true)} МСК`} aria-label="Выбрать время в датасете" style={{ background: `linear-gradient(to right, #1766ef ${(time - startTime) / (endTime - startTime) * 100}%, #dce5f2 0)` }} /><span className="timeline-end">{formatTime(endTime)}</span><label className="speed-control">Скорость <select value={speed} onChange={event => setSpeed(Number(event.target.value))} aria-label="Скорость воспроизведения">{playbackSpeeds.map(value => <option key={value} value={value}>{value}×</option>)}</select><ChevronDown size={13} /></label></div>
      <div className="workspace">
        <section className="primary-panel">
          {view === 'map' ? <TransportMap vehicles={vehicles} selected={selected} selectedId={selectedId} selectedStatus={selected ? statusOf(selected) : 'unknown'} onSelect={select} showRoutes={showRoutes} setShowRoutes={setShowRoutes} collapsed={monitorCollapsed} network={network} /> : <DataView view={view} vehicles={vehicles} select={select} />}
        </section>
        {monitorCollapsed && <button className="restore-monitor" onClick={() => setMonitorCollapsed(false)} title="Открыть мониторинг" aria-label="Открыть мониторинг"><PanelRightOpen size={19} /></button>}
        {monitorCollapsed ? null : <aside className="right-panel">
          {selectedId ? <>
            <button className="collapse-detail" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг" aria-label="Свернуть мониторинг"><PanelRightClose size={18} /></button>
            {selected ? <VehicleDetail vehicle={selected} time={time} onBack={() => setSelectedId(null)} /> : <InactiveRouteDetail id={selectedId} onBack={() => setSelectedId(null)} />}
          </> : <>
            <div className="panel-header">
              <button className="monitor-heading" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг"><span className="eyebrow">МОНИТОРИНГ</span><span className="monitor-title">Транспорт на линии <span>{vehicles.length}</span></span></button>
              <button className="panel-icon-button" title="Свернуть мониторинг" aria-label="Свернуть мониторинг" onClick={() => setMonitorCollapsed(true)}><PanelRightClose size={19} /></button>
            </div>
            <div className="search-box"><Search size={18} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Номер ТС или остановка" aria-label="Поиск транспорта" />{search && <button aria-label="Очистить поиск" onClick={() => setSearch('')}><X size={16} /></button>}</div>
            <div className="filter-row"><button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>Все</button><button className={filter === 'deviation' ? 'active' : ''} onClick={() => setFilter('deviation')}>Отклонения</button><button className={filter === 'risk' ? 'active' : ''} onClick={() => setFilter('risk')}>Риск</button></div>
            <div className="vehicle-list">{filtered.length ? filtered.map(vehicle => <VehicleRow key={vehicle.id} vehicle={vehicle} active={false} onClick={() => setSelectedId(vehicle.id)} />) : <div className="empty-state">Транспорт не найден</div>}</div>
            <div className="panel-footer"><span className="footer-dot" /> {filtered.length} из {vehicles.length} ТС · {formatTime(time)} МСК</div>
          </>}
        </aside>}
      </div>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
