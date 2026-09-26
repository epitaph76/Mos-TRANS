import React, { useCallback, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, AlertTriangle, ArrowLeft, BarChart3, BusFront, CalendarDays,
  ChevronDown, ChevronRight, Clock3, Map as MapIcon,
  MapPin, Menu, Route, Search,
  X, Play, Pause, PanelRightOpen, PanelRightClose,
} from 'lucide-react'
import TransportMap from './TransportMap'
import plan from './data/plan.json'
import demoPlan from './data/demo_plan.json'
import { formatTime, stopSeconds, type SharedNetwork } from './timeline'
import './style.css'
import './detail.css'

type Stop = { id: string; time: string; lon: number; lat: number; name: string }
type Vehicle = {
  id: string; position: [number, number]; speed: number;
  heading: number | null; gpsAgeMin: number;
  stops: Stop[]; nextStop: Stop | null; estimateSeconds: number | null;
  forecastTime: string | null; forecastStopId: string | null; sampleId: string | null;
  forecastStop: Stop | null; probability: number | null; stale: boolean;
  reason: string | null; recommendation: string | null; section: string | null;
  source: string; doorStatus: string | null;
  currentDeviationSeconds: number | null;
}
type View = 'map' | 'schedule' | 'routes' | 'vehicles' | 'events' | 'analytics'
type Filter = 'all' | 'normal' | 'deviation' | 'risk'
type Status = 'normal' | 'minor' | 'delay' | 'critical' | 'early' | 'unknown'

type Mode = 'replay' | 'demo' | 'live'
const initialParams = new URLSearchParams(window.location.search)
const initialMode: Mode = initialParams.get('mode') === 'live' ? 'live' : initialParams.get('demo') === '1' ? 'demo' : 'replay'
const initialDemo = initialMode === 'demo'
const initialAutoplay = initialParams.get('autoplay') === '1'
const playbackSpeeds = [1, 15, 60, 300, 900]

function normalizeVehicle(value: Partial<Vehicle> & { id: string; position?: [number, number] | null }): Vehicle | null {
  if (!value.position || value.position.some(point => point === null || !Number.isFinite(point))) return null
  return {
    id: value.id, position: value.position, speed: value.speed ?? 0,
    heading: value.heading ?? null, gpsAgeMin: value.gpsAgeMin ?? 0,
    stops: value.stops ?? [], nextStop: value.nextStop ?? null,
    estimateSeconds: value.estimateSeconds ?? null, forecastTime: value.forecastTime ?? null,
    forecastStopId: value.forecastStopId ?? null, sampleId: value.sampleId ?? null,
    forecastStop: value.forecastStop ?? null, probability: value.probability ?? null,
    stale: value.stale ?? false, reason: value.reason ?? null,
    recommendation: value.recommendation ?? null, section: value.section ?? null,
    source: value.source ?? 'NDTP', doorStatus: value.doorStatus ?? null,
    currentDeviationSeconds: value.currentDeviationSeconds ?? null,
  }
}

function statusOf(vehicle: Vehicle): Status {
  if (vehicle.probability === null) return 'unknown'
  if (vehicle.probability >= 0.6) return 'critical'
  if (vehicle.probability >= 0.3) return 'minor'
  return 'normal'
}

const statusLabel: Record<Status, string> = {
  normal: 'Низкий риск', minor: 'Средний риск', delay: 'Высокий риск',
  critical: 'Высокий риск', early: 'Опережение', unknown: 'Без прогноза',
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
      <span className="row-meta"><span><Clock3 size={14} /> {vehicle.nextStop ? timeOnly(vehicle.nextStop.time) : '—'}</span><span><Activity size={14} /> {vehicle.speed} км/ч</span><span className={status !== 'unknown' ? `deviation ${status}` : ''}>{vehicle.probability === null ? '—' : `${Math.round(vehicle.probability * 100)}%`}</span></span>
    </span>
    <ChevronRight className="row-chevron" size={18} />
  </button>
}

function VehicleDetail({ vehicle, time, day, demo, onBack }: { vehicle: Vehicle; time: number; day: string; demo: boolean; onBack: () => void }) {
  const status = statusOf(vehicle)
  const [tab, setTab] = useState<'tracking' | 'analytics' | 'details'>('tracking')
  const allStops = vehicle.stops
    .map(stop => ({ stop, timestamp: stopSeconds(stop.time, day) }))
    .sort((a, b) => a.timestamp - b.timestamp)
  const stopEvents = allStops
  const stopPosition = (index: number) => stopEvents.length <= 1 ? 50 : index / (stopEvents.length - 1) * 100
  const nextPlannedIndex = stopEvents.findIndex(event => event.timestamp > time)
  const previousPlannedIndex = nextPlannedIndex === -1 ? stopEvents.length - 1 : nextPlannedIndex - 1
  const plannedFraction = previousPlannedIndex < 0 || nextPlannedIndex < 0 ? 0 :
    Math.max(0, Math.min(1, (time - stopEvents[previousPlannedIndex].timestamp) /
      Math.max(1, stopEvents[nextPlannedIndex].timestamp - stopEvents[previousPlannedIndex].timestamp)))
  const nextObservedIndex = demo ? (vehicle.nextStop ? stopEvents.findIndex(event => event.stop.id === vehicle.nextStop?.id) : stopEvents.length) : -1
  const cursorPercent = stopPosition(Math.max(0, previousPlannedIndex) + plannedFraction)
  const rulerLabels = [0, 0.25, 0.5, 0.75, 1].map(fraction =>
    stopEvents.length ? timeOnly(stopEvents[Math.round(fraction * (stopEvents.length - 1))].stop.time) : '—')
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
      <div className="tracking-ruler" aria-label="Плановые остановки по порядку; расстояние между метками на схеме одинаковое">
        {rulerLabels.map((label, index) => <span key={index}>{label}</span>)}
        <div className="tracking-track"><i className="tracking-current" style={{ left: `${cursorPercent}%` }} />{stopEvents.map(({ stop, timestamp }, index) => <i key={`${stop.id}-${index}`} className={`tracking-stop ${demo && index < nextObservedIndex ? 'reached' : demo && timestamp < time ? 'overdue' : ''}`} style={{ left: `${stopPosition(index)}%` }} title={`${timeOnly(stop.time)} · ${stopName(stop.name)}`} />)}</div>
      </div>
      <p className="tracking-legend">{demo ? 'Синий — достигнута по GPS · красный — плановое время прошло, ТС ещё не прибыло' : 'Метки — остановки по порядку, синяя линия — плановое положение во время среза'}</p>
      {vehicle.nextStop && <div className="next-stop-feature"><MapPin size={18} /><div><span>{demo ? 'Следующая по GPS' : 'Следующая по расписанию'}</span><strong>{stopName(vehicle.nextStop.name)}</strong><small>План · {timeOnly(vehicle.nextStop.time)}</small></div></div>}
      <div className="stop-event-list">{stopEvents.length ? stopEvents.map(({ stop, timestamp }, index) => {
        const reached = demo && index < nextObservedIndex
        const overdue = demo && !reached && timestamp < time
        const state = demo ? reached ? 'Достигнута по GPS' : overdue ? 'Ожидается с опозданием' : 'Впереди по плану' : timestamp <= time ? 'Плановое время прошло' : 'Далее по графику'
        const intervalMinutes = index ? Math.round((timestamp - stopEvents[index - 1].timestamp) / 60) : 0
        const interval = index ? intervalMinutes >= 10 ? `Разрыв в расписании ${intervalMinutes} мин · ` : intervalMinutes === 0 ? 'В ту же минуту · ' : `Интервал ${intervalMinutes} мин · ` : ''
        return <div className={`stop-event ${intervalMinutes >= 10 ? 'schedule-gap' : ''}`} key={`${stop.id}-${index}`}><span className={`stop-event-dot ${reached || !demo && timestamp <= time ? 'past' : ''} ${overdue ? 'overdue' : ''}`} /><time>{timeOnly(stop.time)}</time><span>{stopName(stop.name)}</span><small>{interval}{state}</small></div>
      }) : <p className="empty-stop-events">Нет плановых остановок для выбранного времени.</p>}</div>
    </div>}
    {tab === 'analytics' && <div className="detail-tab-content" role="tabpanel">
      <div className="prediction-block"><span className="eyebrow">Прогноз на 10–15 минут</span><strong className={status}>{minutes(vehicle.estimateSeconds)}</strong><p>{vehicle.probability === null ? 'В выбранный момент нет прогнозной точки с доступной моделью.' : `Вероятность опоздания более чем на 2 минуты: ${Math.round(vehicle.probability * 100)}%.`}</p></div>
      {vehicle.probability !== null && <div className="detail-section"><h3>Карточка риска</h3><div className="metric-line"><span>Целевая остановка</span><strong>{vehicle.forecastStop ? stopName(vehicle.forecastStop.name) : vehicle.forecastStopId}</strong></div><div className="metric-line"><span>Плановое прибытие</span><strong>{timeOnly(vehicle.forecastTime)}</strong></div><div className="metric-line"><span>Участок маршрута</span><strong>{vehicle.section ?? 'Участок не определён'}</strong></div><div className="metric-line"><span>Предполагаемая причина</span><strong>{vehicle.reason ?? 'Причина не определена'}</strong></div><div className="metric-line"><span>Действие</span><strong>{vehicle.recommendation ?? 'Наблюдать'}</strong></div></div>}
      <div className="detail-section"><h3>Показатели движения</h3><div className="metric-line"><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div><div className="metric-line"><span>Текущее отклонение</span><strong>{minutes(vehicle.currentDeviationSeconds)}</strong></div><div className="metric-line"><span>Прогноз отклонения</span><strong>{minutes(vehicle.estimateSeconds)}</strong></div></div>
    </div>}
    {tab === 'details' && <div className="detail-tab-content" role="tabpanel">
      <div className="detail-section"><h3>Телеметрия</h3><div className="metric-line"><span>Последний GPS</span><strong>{vehicle.gpsAgeMin < 1 ? 'менее минуты назад' : `${Math.round(vehicle.gpsAgeMin)} мин назад`}</strong></div><div className="metric-line"><span>Координаты</span><strong>GPS-трек NDTP</strong></div><div className="metric-line"><span>Двери</span><strong>{vehicle.doorStatus ?? 'Нет данных в историческом CSV'}</strong></div></div>
      {vehicle.forecastTime && <div className="detail-section"><h3>Прогнозная точка</h3><div className="metric-line"><span>Целевая остановка</span><strong>{timeOnly(vehicle.forecastTime)}</strong></div><div className="metric-line"><span>ID точки</span><strong>{vehicle.sampleId}</strong></div></div>}
      <div className="detail-section"><h3>Данные</h3><div className="metric-line"><span>Источник</span><strong>{vehicle.source}</strong></div><div className="metric-line"><span>Время среза</span><strong>{formatTime(time)} МСК</strong></div><div className="metric-line"><span>Свежесть GPS</span><strong>{vehicle.stale ? 'Последняя позиция устарела' : 'Актуальная позиция'}</strong></div></div>
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

function LiveDetail({ vehicle, onBack }: { vehicle: Vehicle; onBack: () => void }) {
  return <div className="detail-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К списку ТС</button>
    <div className="vehicle-card-heading"><div className="vehicle-card-title"><span className="detail-bus unknown"><BusFront size={25} /></span><div><h2>Устройство {vehicle.id}</h2><span className="detail-kicker">Живой поток NDTP</span></div></div></div>
    <div className="prediction-block"><span className="eyebrow">Прогноз недоступен</span><p>Эмулятор не привязан к плановому расписанию. Его случайную траекторию нельзя оценивать как рейс.</p></div>
    <div className="detail-section"><h3>Телеметрия</h3><div className="metric-line"><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div><div className="metric-line"><span>Последний GPS</span><strong>{Math.round(vehicle.gpsAgeMin * 60)} с назад</strong></div><div className="metric-line"><span>Координаты</span><strong>{vehicle.position[0].toFixed(5)}, {vehicle.position[1].toFixed(5)}</strong></div><div className="metric-line"><span>Двери</span><strong>{vehicle.doorStatus ?? 'Нет данных от устройства'}</strong></div><div className="metric-line"><span>Источник</span><strong>{vehicle.source}</strong></div></div>
  </div>
}

function DataView({ view, vehicles, select }: { view: View; vehicles: Vehicle[]; select: (id: string) => void }) {
  if (view === 'schedule') {
    const rows = vehicles.filter(v => v.nextStop).sort((a, b) => (a.nextStop?.time || '').localeCompare(b.nextStop?.time || ''))
    return <div className="data-view"><div className="data-intro"><h2>Ближайшие прибытия</h2><p>Плановое расписание для транспорта на линии</p></div><div className="data-table"><div className="table-header"><span>Время</span><span>Транспорт</span><span>Остановка</span><span>Состояние</span></div>{rows.map(v => <button key={v.id} className="table-row" onClick={() => select(v.id)}><strong>{timeOnly(v.nextStop!.time)}</strong><span>ТС {v.id}</span><span>{stopName(v.nextStop!.name)}</span><span className={`status-badge ${statusOf(v)}`}>{statusLabel[statusOf(v)]}</span></button>)}</div></div>
  }
  if (view === 'analytics') {
    const tracked = vehicles.filter(v => v.estimateSeconds !== null)
    return <div className="data-view"><div className="data-intro"><h2>Аналитика движения</h2><p>Вероятность задержки более 2 минут на целевой остановке</p></div><div className="analytics-grid"><div className="analytic-tile"><span>На линии</span><strong>{vehicles.length}</strong><small>ТС с полученной телеметрией</small></div><div className="analytic-tile"><span>С прогнозом</span><strong>{tracked.length}</strong><small>Цель через 10–15 минут</small></div><div className="analytic-tile"><span>Высокий риск</span><strong>{tracked.filter(v => statusOf(v) === 'critical').length}</strong><small>Вероятность от 60%</small></div></div><h3 className="view-subtitle">Транспорт с прогнозом</h3><div className="data-table compact">{tracked.map(v => <button key={v.id} className="analytic-row" onClick={() => select(v.id)}><span className={`tiny-dot ${statusOf(v)}`} /> ТС {v.id}<strong>{Math.round((v.probability ?? 0) * 100)}% · {minutes(v.estimateSeconds)}</strong><ChevronRight size={17} /></button>)}</div></div>
  }
  const title = view === 'routes' ? 'Маршрутная сеть' : view === 'events' ? 'События' : 'Транспорт на линии'
  const shown = view === 'events' ? vehicles.filter(v => ['minor', 'critical'].includes(statusOf(v))) : vehicles
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
  const [time, setTime] = useState(initialDemo ? demoPlan.start : plan.start)
  const [playing, setPlaying] = useState(initialDemo && initialAutoplay)
  const [mode, setMode] = useState<Mode>(initialMode)
  const activePlan = mode === 'demo' ? demoPlan : plan
  const startTime = activePlan.start
  const endTime = activePlan.end
  const network = activePlan.network as SharedNetwork
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const [modelStatus, setModelStatus] = useState('ready')
  const timeRef = useRef(time)
  timeRef.current = time
  const [speed, setSpeed] = useState(initialDemo ? 15 : 60)
  const [monitorCollapsed, setMonitorCollapsed] = useState(false)
  const [view, setView] = useState<View>('map')
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(initialDemo ? '130389' : null)
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
  }, [playing, speed, endTime])
  useEffect(() => {
    if (time >= endTime) setPlaying(false)
  }, [time, endTime])
  useEffect(() => {
    let mounted = true
    let busy = false
    let lastRequestedTime = -1
    const load = async () => {
      if (busy) return
      const requestedTime = Math.floor(timeRef.current)
      if (mode !== 'live' && requestedTime === lastRequestedTime) return
      busy = true
      try {
        const url = mode === 'live' ? '/api/live/snapshot' : `/api/${mode}/snapshot?at=${requestedTime}`
        const response = await fetch(url)
        if (!response.ok) throw new Error(`API ${response.status}`)
        const state = await response.json()
        if (mounted) {
          if (state.modelStatus !== 'unavailable') lastRequestedTime = requestedTime
          setVehicles((state.vehicles as (Partial<Vehicle> & { id: string })[]).map(normalizeVehicle).filter((v): v is Vehicle => v !== null))
          setModelStatus(state.modelStatus ?? 'ready')
          setApiError(null)
        }
      } catch (error) {
        if (mounted) { setApiError(error instanceof Error ? error.message : 'Нет связи с API'); setModelStatus('unavailable') }
      } finally { busy = false }
    }
    void load()
    const timer = window.setInterval(() => { void load() }, mode === 'live' ? 1000 : 400)
    return () => { mounted = false; window.clearInterval(timer) }
  }, [mode])
  const selected = vehicles.find(v => v.id === selectedId) ?? null
  const filtered = vehicles.filter(v => {
    const status = statusOf(v)
    if (filter === 'normal' && status !== 'normal') return false
    if (filter === 'deviation' && (v.estimateSeconds === null || Math.abs(v.estimateSeconds) < 60)) return false
    if (filter === 'risk' && !['minor', 'critical'].includes(status)) return false
    return `${v.id} ${v.nextStop?.name ?? ''}`.toLocaleLowerCase('ru').includes(search.toLocaleLowerCase('ru'))
  })
  const countRisk = vehicles.filter(v => statusOf(v) === 'critical').length
  const switchMode = (next: Mode) => {
    const url = new URL(window.location.href)
    url.searchParams.delete('autoplay')
    if (next === 'demo') url.searchParams.set('demo', '1')
    else url.searchParams.delete('demo')
    if (next === 'live') url.searchParams.set('mode', 'live')
    else url.searchParams.delete('mode')
    window.history.replaceState(null, '', url)
    setMode(next)
    setPlaying(false)
    setTime((next === 'demo' ? demoPlan : plan).start)
    setSelectedId(next === 'demo' ? '130389' : null)
    setSpeed(next === 'demo' ? 15 : 60)
  }
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
      <header className="topbar"><div className="topbar-title"><button className="mobile-menu" aria-label="Открыть меню" onClick={() => setMobileNav(true)}><Menu size={22} /></button><h1>{view === 'map' ? 'Карта' : navItems.find(item => item.id === view)?.label}</h1></div><div className="topbar-actions"><div className="mode-picker" aria-label="Источник данных"><button className={mode === 'demo' ? 'active' : ''} onClick={() => switchMode('demo')}>Учебный рейс</button><button className={mode === 'replay' ? 'active' : ''} onClick={() => switchMode('replay')}>История</button><button className={mode === 'live' ? 'active' : ''} onClick={() => switchMode('live')}>NDTP</button></div><div className="date-chip"><CalendarDays size={16} /> {mode === 'live' ? 'Прямой эфир' : activePlan.date}</div><div className="time-select"><Clock3 size={17} />{mode === 'live' ? 'сейчас' : `${formatTime(time, true)} МСК`}</div></div></header>
      {mode === 'demo' && <div className="demo-banner"><strong>Учебный рейс 130389</strong><span>09:00–09:15 по графику · 09:15–09:22 вынужденный простой · затем замедление и накопление опоздания. Прогнозы ниже выдают настоящие модели на синтетических признаках.</span></div>}
      {apiError && <div className="service-banner">Нет связи с backend: {apiError}. Показано последнее полученное состояние.</div>}
      {modelStatus === 'unavailable' && !apiError && <div className="service-banner">ML-сервис недоступен: положение ТС обновляется, новые прогнозы временно не рассчитываются.</div>}
      {mode !== 'live' && <div className="timeline-bar"><button className="play-button" onClick={() => { if (time >= endTime) setTime(startTime); setPlaying(!playing || time >= endTime) }} aria-label={playing ? 'Пауза' : 'Воспроизвести'} title={playing ? 'Пауза' : 'Воспроизвести'}>{playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}</button><span className="timeline-time">{formatTime(time, true)}</span><input className="time-slider" type="range" min={startTime} max={endTime} step="1" value={time} onChange={event => { setPlaying(false); setTime(Number(event.target.value)) }} aria-valuetext={`${formatTime(time, true)} МСК`} aria-label="Выбрать время в датасете" style={{ background: `linear-gradient(to right, #1766ef ${(time - startTime) / (endTime - startTime) * 100}%, #dce5f2 0)` }} /><span className="timeline-end">{formatTime(endTime)}</span><label className="speed-control">Скорость <select value={speed} onChange={event => setSpeed(Number(event.target.value))} aria-label="Скорость воспроизведения">{playbackSpeeds.map(value => <option key={value} value={value}>{value}×</option>)}</select><ChevronDown size={13} /></label></div>}
      <div className="workspace">
        <section className="primary-panel">
          {view === 'map' ? <TransportMap vehicles={vehicles} selected={selected} selectedId={selectedId} selectedStatus={selected ? statusOf(selected) : 'unknown'} onSelect={select} showRoutes={showRoutes && mode !== 'live'} setShowRoutes={setShowRoutes} collapsed={monitorCollapsed} network={network} /> : <DataView view={view} vehicles={vehicles} select={select} />}
        </section>
        {monitorCollapsed && <button className="restore-monitor" onClick={() => setMonitorCollapsed(false)} title="Открыть мониторинг" aria-label="Открыть мониторинг"><PanelRightOpen size={19} /></button>}
        {monitorCollapsed ? null : <aside className="right-panel">
          {selectedId ? <>
            <button className="collapse-detail" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг" aria-label="Свернуть мониторинг"><PanelRightClose size={18} /></button>
            {selected ? mode === 'live' ? <LiveDetail vehicle={selected} onBack={() => setSelectedId(null)} /> : <VehicleDetail vehicle={selected} time={time} day={activePlan.date} demo={mode === 'demo'} onBack={() => setSelectedId(null)} /> : <InactiveRouteDetail id={selectedId} onBack={() => setSelectedId(null)} />}
          </> : <>
            <div className="panel-header">
              <button className="monitor-heading" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг"><span className="eyebrow">МОНИТОРИНГ</span><span className="monitor-title">Транспорт на линии <span>{vehicles.length}</span></span></button>
              <button className="panel-icon-button" title="Свернуть мониторинг" aria-label="Свернуть мониторинг" onClick={() => setMonitorCollapsed(true)}><PanelRightClose size={19} /></button>
            </div>
            <div className="search-box"><Search size={18} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Номер ТС или остановка" aria-label="Поиск транспорта" />{search && <button aria-label="Очистить поиск" onClick={() => setSearch('')}><X size={16} /></button>}</div>
            <div className="filter-row"><button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>Все</button><button className={filter === 'deviation' ? 'active' : ''} onClick={() => setFilter('deviation')}>Отклонения</button><button className={filter === 'risk' ? 'active' : ''} onClick={() => setFilter('risk')}>Риск</button></div>
            <div className="vehicle-list">{filtered.length ? filtered.map(vehicle => <VehicleRow key={vehicle.id} vehicle={vehicle} active={false} onClick={() => setSelectedId(vehicle.id)} />) : <div className="empty-state">Транспорт не найден</div>}</div>
            <div className="panel-footer"><span className="footer-dot" /> {filtered.length} из {vehicles.length} ТС · {mode === 'live' ? 'прямой эфир' : `${formatTime(time)} МСК`}</div>
          </>}
        </aside>}
      </div>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
