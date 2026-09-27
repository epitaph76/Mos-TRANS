import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, AlertTriangle, ArrowLeft, BarChart3, BusFront, CalendarDays,
  ChevronDown, ChevronRight, Clock3, Map as MapIcon,
  MapPin, Menu, Route, Search,
  X, Play, Pause, PanelRightOpen, PanelRightClose,
} from 'lucide-react'
import TransportMap from './TransportMap'
import plan from './data/plan.json'
import roadRoutes from './data/road_routes.json'
import { describeRoutes, type RouteGroup } from './routes'
import { estimateBetweenFixes, formatTime, matchRoadFix, pointOnRoadRoute, stopSeconds,
  type RoadMatch, type SharedNetwork } from './timeline'
import { riskStatus, type RiskStatus } from './risk'
import DelayExplanation, { type DelayExplanationData } from './DelayExplanation'
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
  forecastAvailability: 'ready' | 'no_point' | 'no_target' | 'no_schedule' | 'bad_gps' | 'off_route' | 'late_packet' | 'stale_gps' | 'ml_unavailable' | 'pending';
  forecastGeneratedAt: string | null; forecastPacketId: string | null;
  nearestForecastPointAt: string | null;
  nearestForecastPointDirection: 'next' | 'previous' | null;
  gpsEventTime: string | null;
  positionMethod: 'gps' | 'route' | 'heading';
  route?: string; tripId?: string; direction?: string;
  routeKey?: string;
  routeProgress?: number;
  forecastMethod?: string | null; stopEvents?: { stopId: string; name: string; time: string; delaySeconds: number }[];
  predictedArrivalAt?: string | null;
}
type View = 'map' | 'vehicles' | 'events' | 'analytics'
type Filter = 'all' | 'normal' | 'deviation' | 'risk'
type Status = RiskStatus

type Mode = 'replay' | 'live'
const initialParams = new URLSearchParams(window.location.search)
const initialMode: Mode = initialParams.get('mode') === 'live' ? 'live' : 'replay'
const initialAutoplay = initialParams.get('autoplay') === '1'
const playbackSpeeds = [1, 3, 15, 60, 300, 900]
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
    route: value.route, tripId: value.tripId, direction: value.direction,
    forecastMethod: value.forecastMethod, stopEvents: value.stopEvents,
    predictedArrivalAt: value.predictedArrivalAt ?? null,
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

function timeOnly(value: string | null) {
  return value ? value.slice(11, 16) : '—'
}

function moscowTime(value: string | null) {
  return value ? new Date(value).toLocaleTimeString('ru-RU', {
    timeZone: 'Europe/Moscow', hour: '2-digit', minute: '2-digit',
  }) : '—'
}

function VehicleRow({ vehicle, active, live, onClick }: { vehicle: Vehicle; active: boolean; live: boolean; onClick: () => void }) {
  const status = statusOf(vehicle)
  return <button className={`vehicle-row ${active ? 'active' : ''}`} onClick={onClick}>
    <span className={`row-icon ${status}`}><BusFront size={21} strokeWidth={2.1} /></span>
    <span className="row-content">
      <span className="row-heading"><strong>ТС {vehicle.id}</strong><span className={`status-badge ${status}`}>{statusLabelOf(vehicle)}</span></span>
      <span className="row-location"><MapPin size={13} /> {vehicle.nextStop ? stopName(vehicle.nextStop.name) : 'Остановка не указана'}</span>
      <span className="row-meta"><span><Clock3 size={14} /> {vehicle.nextStop ? live ? moscowTime(vehicle.nextStop.time) : timeOnly(vehicle.nextStop.time) : '—'}</span><span><Activity size={14} /> {vehicle.speed} км/ч</span><span className={status !== 'unknown' ? `deviation ${status}` : ''}>{vehicle.route && vehicle.currentDeviationSeconds !== null ? minutes(vehicle.currentDeviationSeconds) : vehicle.probability === null ? '—' : `${Math.round(vehicle.probability * 100)}%`}</span></span>
    </span>
    <ChevronRight className="row-chevron" size={18} />
  </button>
}

function ScheduleTimeline({ stops, nextStop, now, stopTime, clock, momentLabel,
  passages = [], targetStopId, live = false }: {
  stops: Stop[]; nextStop: Stop | null; now: number;
  stopTime: (stop: Stop) => number; clock: (value: string | null) => string;
  momentLabel: string; passages?: NonNullable<Vehicle['stopEvents']>;
  targetStopId?: string | null; live?: boolean;
}) {
  const events = stops.map(stop => ({ stop, timestamp: stopTime(stop) }))
    .sort((a, b) => a.timestamp - b.timestamp)
  const passed = new Map(passages.map(event => [event.stopId, event]))
  const stopPosition = (index: number) => events.length <= 1 ? 50 : index / (events.length - 1) * 100
  const nextIndex = events.findIndex(event => event.timestamp > now)
  const previousIndex = nextIndex === -1 ? events.length - 1 : nextIndex - 1
  const fraction = previousIndex < 0 || nextIndex < 0 ? 0 :
    Math.max(0, Math.min(1, (now - events[previousIndex].timestamp) /
      Math.max(1, events[nextIndex].timestamp - events[previousIndex].timestamp)))
  const cursorPercent = stopPosition(Math.max(0, previousIndex) + fraction)
  const labels = [0, 0.25, 0.5, 0.75, 1].map(value =>
    events.length ? clock(events[Math.round(value * (events.length - 1))].stop.time) : '—')
  return <div className="detail-tab-content" role="tabpanel">
    <div className="tracking-heading"><strong>Остановки по расписанию</strong><span>{momentLabel} МСК</span></div>
    {events.length > 0 && <>
      <div className="tracking-ruler" aria-label="Плановые остановки по порядку; расстояние между метками на схеме одинаковое">
        {labels.map((label, index) => <span key={index}>{label}</span>)}
        <div className="tracking-track"><i className="tracking-current" style={{ left: `${cursorPercent}%` }} />{events.map(({ stop, timestamp }, index) => <i key={`${stop.id}-${index}`} className={`tracking-stop ${timestamp <= now ? 'past' : ''} ${passed.has(stop.id) ? 'observed' : ''} ${stop.id === targetStopId ? 'target' : ''}`} style={{ left: `${stopPosition(index)}%` }} title={`${clock(stop.time)} · ${stopName(stop.name)}`} />)}</div>
      </div>
      <p className="tracking-legend">Синие — впереди по расписанию, серые — плановое время уже прошло. Вертикальная линия — {live ? 'момент GPS' : 'выбранный момент'}.{passages.length ? ' Зелёные — проход по GPS.' : ''}</p>
    </>}
    {nextStop && <div className="next-stop-feature"><MapPin size={18} /><div><span>Следующая {live ? 'по маршруту' : 'по расписанию'}</span><strong>{stopName(nextStop.name)}</strong><small>План · {clock(nextStop.time)}</small></div></div>}
    <div className="stop-event-list">{events.length ? events.map(({ stop, timestamp }, index) => {
      const passage = passed.get(stop.id)
      const state = passage ? `Проход по GPS ${clock(passage.time)} · отклонение ${minutes(passage.delaySeconds)}`
        : timestamp <= now ? 'Плановое время прошло' : 'Далее по графику'
      const intervalMinutes = index ? Math.round((timestamp - events[index - 1].timestamp) / 60) : 0
      const interval = index ? intervalMinutes >= 10 ? `Разрыв в расписании ${intervalMinutes} мин · ` : intervalMinutes === 0 ? 'В ту же минуту · ' : `Интервал ${intervalMinutes} мин · ` : ''
      return <div className={`stop-event ${intervalMinutes >= 10 ? 'schedule-gap' : ''} ${stop.id === targetStopId ? 'forecast-target' : ''}`} key={`${stop.id}-${index}`}><span className={`stop-event-dot ${timestamp <= now ? 'past' : ''} ${passage ? 'observed' : ''}`} /><time>{clock(stop.time)}</time><span>{stopName(stop.name)}</span><small>{interval}{stop.id === targetStopId ? 'Цель прогноза · ' : ''}{state}</small></div>
    }) : <p className="empty-stop-events">{live ? 'Для этого рейса нет плановых остановок.' : 'Нет плановых остановок для выбранного времени.'}</p>}</div>
  </div>
}

function VehicleDetail({ vehicle, time, day, onBack, onJumpToForecast }: { vehicle: Vehicle; time: number; day: string; onBack: () => void; onJumpToForecast: (seconds: number) => void }) {
  const status = statusOf(vehicle)
  const [tab, setTab] = useState<'tracking' | 'analytics' | 'details'>('tracking')
  const nearestPointLabel = timeOnly(vehicle.nearestForecastPointAt)
  const unavailableReason = vehicle.forecastAvailability === 'ml_unavailable'
    ? 'Прогнозная точка есть, но ML-сервис сейчас недоступен.'
    : vehicle.forecastAvailability === 'pending'
      ? 'Прогнозная точка есть, ответ модели пока не получен.'
      : vehicle.forecastAvailability === 'no_target'
        ? 'По расписанию нет остановки через 10–15 минут.'
        : vehicle.forecastAvailability === 'bad_gps'
          ? 'Нет свежей достоверной координаты для расчёта прогноза.'
          : vehicle.forecastAvailability === 'off_route'
            ? 'GPS не удалось надёжно сопоставить с плановым маршрутом.'
            : vehicle.forecastAvailability === 'late_packet'
              ? 'Получен запоздавший пакет; ожидается новая координата.'
            : vehicle.forecastAvailability === 'stale_gps'
              ? 'Новых GPS-пакетов не поступало более двух минут.'
      : vehicle.nearestForecastPointAt
        ? `В ${formatTime(time)} для этого ТС нет прогнозной точки. Ближайшая ${vehicle.nearestForecastPointDirection === 'next' ? 'следующая' : 'предыдущая'} — в ${nearestPointLabel}. Исторический датасет размечен отдельными срезами.`
        : 'Для этого ТС в архиве нет прогнозных точек.'
  return <div className="detail-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К списку ТС</button>
    <div className="vehicle-card-heading">
      <div className="vehicle-card-title"><span className={`detail-bus ${status}`}><BusFront size={25} /></span><div><h2>ТС {vehicle.id}</h2><span className="detail-kicker">Московский наземный транспорт</span></div></div>
      <span className={`detail-status ${status}`}><span className="status-dot" />{statusLabelOf(vehicle)}</span>
    </div>
    <div className="vehicle-meta-grid">
      <div><span>Ближайшая остановка</span><strong>{vehicle.nextStop ? stopName(vehicle.nextStop.name) : 'Нет данных'}</strong></div>
      <div><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div>
      <div><span>Последний GPS</span><strong>{vehicle.gpsAgeMin < 1 ? 'менее минуты' : `${Math.round(vehicle.gpsAgeMin)} мин`}</strong></div>
    </div>
    <div className="detail-forecast-top"><div className="prediction-block"><span className="eyebrow">{vehicle.forecastStale ? 'Последний прогноз задержки' : 'Прогноз задержки через 10–15 минут'}</span><strong className={status}>{minutes(vehicle.estimateSeconds)}</strong><p>{vehicle.probability === null ? unavailableReason : `Вероятность опоздания более чем на 2 минуты: ${Math.round(vehicle.probability * 100)}%.`}</p>{vehicle.probability !== null && vehicle.forecastGeneratedAt && <p>{vehicle.forecastStale ? 'Рассчитан' : 'Обновлено по GPS'} в {timeOnly(vehicle.forecastGeneratedAt)}</p>}{vehicle.forecastStale && <p className="forecast-stale-note">Свежих данных для нового расчёта нет. Показан последний доступный прогноз для остановки в {timeOnly(vehicle.forecastTime)}{vehicle.forecastExpired ? ' (её плановое время прошло)' : ''}.</p>}{['minor', 'critical'].includes(status) && vehicle.estimateSeconds !== null && vehicle.estimateSeconds > 0 && vehicle.delayExplanation && <DelayExplanation data={vehicle.delayExplanation} />}{vehicle.probability === null && vehicle.forecastAvailability === 'no_point' && vehicle.nearestForecastPointAt && <button className="forecast-jump" onClick={() => onJumpToForecast(stopSeconds(vehicle.nearestForecastPointAt!, day))}>Перейти к точке {nearestPointLabel}</button>}</div></div>
    <div className="detail-tabs" role="tablist" aria-label="Информация о транспортном средстве">
      <button role="tab" aria-selected={tab === 'tracking'} className={tab === 'tracking' ? 'active' : ''} onClick={() => setTab('tracking')}>Маршрут</button>
      <button role="tab" aria-selected={tab === 'analytics'} className={tab === 'analytics' ? 'active' : ''} onClick={() => setTab('analytics')}>Аналитика</button>
      <button role="tab" aria-selected={tab === 'details'} className={tab === 'details' ? 'active' : ''} onClick={() => setTab('details')}>Данные</button>
    </div>
    {tab === 'tracking' && <ScheduleTimeline stops={vehicle.stops} nextStop={vehicle.nextStop} now={time}
      stopTime={stop => stopSeconds(stop.time, day)} clock={timeOnly} momentLabel={formatTime(time)} />}
    {tab === 'analytics' && <div className="detail-tab-content" role="tabpanel">
      {vehicle.probability !== null && <div className="detail-section"><h3>Карточка риска</h3><div className="metric-line"><span>Целевая остановка</span><strong>{vehicle.forecastStop ? stopName(vehicle.forecastStop.name) : vehicle.forecastStopId}</strong></div><div className="metric-line"><span>Плановое прибытие</span><strong>{timeOnly(vehicle.forecastTime)}</strong></div><div className="metric-line"><span>Участок маршрута</span><strong>{vehicle.section ?? 'Участок не определён'}</strong></div><div className="metric-line"><span>Наблюдаемый сигнал</span><strong>{vehicle.reason ?? 'Не определён'}</strong></div><div className="metric-line"><span>Действие</span><strong>{vehicle.recommendation ?? 'Наблюдать'}</strong></div></div>}
      <div className="detail-section"><h3>Показатели движения</h3><div className="metric-line"><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div><div className="metric-line"><span>Средняя скорость за 5 минут</span><strong>{vehicle.meanSpeed5m === null ? '—' : `${Math.round(vehicle.meanSpeed5m)} км/ч`}</strong></div><div className="metric-line"><span>Средняя скорость на участке</span><strong>{vehicle.segmentSpeedMeanKmh === null ? '—' : `${Math.round(vehicle.segmentSpeedMeanKmh)} км/ч`}</strong></div><div className="metric-line"><span>Текущий простой</span><strong>{vehicle.stoppedDurationSeconds === null ? '—' : `${Math.round(vehicle.stoppedDurationSeconds)} с`}</strong></div><div className="metric-line"><span>Текущее отклонение</span><strong>{minutes(vehicle.currentDeviationSeconds)}</strong></div><div className="metric-line"><span>Прогноз отклонения</span><strong>{minutes(vehicle.estimateSeconds)}</strong></div></div>
    </div>}
    {tab === 'details' && <div className="detail-tab-content" role="tabpanel">
      <div className="detail-section"><h3>Телеметрия</h3><div className="metric-line"><span>Последний GPS</span><strong>{vehicle.gpsAgeMin < 1 ? 'менее минуты назад' : `${Math.round(vehicle.gpsAgeMin)} мин назад`}</strong></div><div className="metric-line"><span>Положение на карте</span><strong>{vehicle.positionMethod === 'route' ? 'Расчёт по маршруту и последней скорости' : vehicle.positionMethod === 'heading' ? 'Расчёт по курсу и последней скорости' : 'Последний полученный GPS'}</strong></div><div className="metric-line"><span>Двери</span><strong>{vehicle.doorStatus ?? 'Нет данных в историческом CSV'}</strong></div></div>
      {vehicle.forecastTime && <div className="detail-section"><h3>Прогнозная точка</h3><div className="metric-line"><span>Целевая остановка</span><strong>{timeOnly(vehicle.forecastTime)}</strong></div><div className="metric-line"><span>ID точки</span><strong>{vehicle.sampleId}</strong></div></div>}
      <div className="detail-section"><h3>Данные</h3><div className="metric-line"><span>Источник</span><strong>{vehicle.source}</strong></div><div className="metric-line"><span>Время среза</span><strong>{formatTime(time)} МСК</strong></div><div className="metric-line"><span>Свежесть GPS</span><strong>{vehicle.stale ? 'Последняя позиция устарела' : 'Актуальная позиция'}</strong></div></div>
    </div>}
  </div>
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
    <div className="prediction-block"><p>В выбранный момент свежей телеметрии нет. На карте показана траектория по доступным GPS-точкам за день. Выберите другое время на шкале, чтобы увидеть движение ТС.</p></div>
  </div>
}

function LiveDetail({ vehicle, onBack }: { vehicle: Vehicle; onBack: () => void }) {
  const [tab, setTab] = useState<'tracking' | 'analytics' | 'details'>('tracking')
  const status = statusOf(vehicle)
  const hasMlForecast = vehicle.forecastMethod === 'ml' && vehicle.probability !== null &&
    vehicle.estimateSeconds !== null
  const gpsTime = vehicle.gpsEventTime ? Date.parse(vehicle.gpsEventTime) / 1000 : 0
  const unavailableReason = vehicle.forecastAvailability === 'no_schedule'
    ? 'Для этого рейса нет расписания: доступно только положение по GPS.'
    : vehicle.forecastAvailability === 'ml_unavailable'
      ? 'ML-сервис недоступен. Положение и текущее отклонение продолжают обновляться.'
      : vehicle.forecastAvailability === 'no_target'
        ? 'По расписанию пока нет целевой остановки через 10–15 минут.'
        : vehicle.forecastAvailability === 'off_route'
          ? 'GPS пока не удалось сопоставить с плановым маршрутом.'
          : 'Ожидается ответ ML-модели по поступившим GPS-пакетам.'
  return <div className="detail-pane">
    <button className="back-button" onClick={onBack}><ArrowLeft size={17} /> К списку ТС</button>
    <div className="vehicle-card-heading"><div className="vehicle-card-title"><span className={`detail-bus ${status}`}><BusFront size={25} /></span><div><h2>ТС {vehicle.id}</h2><span className="detail-kicker">{vehicle.route ? `Маршрут ${vehicle.route} · ` : ''}NDTP</span></div></div><span className={`detail-status ${status}`}><span className="status-dot" />{statusLabelOf(vehicle)}</span></div>
    <div className="vehicle-meta-grid"><div><span>Ближайшая остановка</span><strong>{vehicle.nextStop ? stopName(vehicle.nextStop.name) : 'Нет данных'}</strong></div><div><span>Скорость</span><strong>{vehicle.speed} км/ч</strong></div><div><span>Последний GPS</span><strong>{vehicle.gpsAgeMin < 1 ? 'менее минуты' : `${Math.round(vehicle.gpsAgeMin)} мин`}</strong></div></div>
    <div className="detail-forecast-top"><div className="prediction-block">
      <span className="eyebrow">Прогноз задержки через 10–15 минут</span>
      <strong className={hasMlForecast ? status : 'unknown'}>{minutes(hasMlForecast ? vehicle.estimateSeconds : null)}</strong>
      <p>{hasMlForecast ? `Вероятность опоздания более чем на 2 минуты: ${Math.round(vehicle.probability! * 100)}%.` : unavailableReason}</p>
      {hasMlForecast && vehicle.forecastGeneratedAt && <p>Обновлено по GPS в {moscowTime(vehicle.forecastGeneratedAt)}</p>}
      {vehicle.forecastStop && <p>Цель: {stopName(vehicle.forecastStop.name)} · план {moscowTime(vehicle.forecastTime)}{hasMlForecast && vehicle.predictedArrivalAt ? ` · прогноз ${moscowTime(vehicle.predictedArrivalAt)}` : ''}</p>}
      {hasMlForecast && vehicle.delayExplanation && <DelayExplanation data={vehicle.delayExplanation} />}
    </div><div className="live-deviation-line"><span>Текущее отклонение по GPS и расписанию</span><strong>{minutes(vehicle.currentDeviationSeconds)}</strong></div></div>
    <div className="detail-tabs" role="tablist" aria-label="Информация о транспортном средстве">
      <button role="tab" aria-selected={tab === 'tracking'} className={tab === 'tracking' ? 'active' : ''} onClick={() => setTab('tracking')}>Маршрут</button>
      <button role="tab" aria-selected={tab === 'analytics'} className={tab === 'analytics' ? 'active' : ''} onClick={() => setTab('analytics')}>Аналитика</button>
      <button role="tab" aria-selected={tab === 'details'} className={tab === 'details' ? 'active' : ''} onClick={() => setTab('details')}>Данные</button>
    </div>
    {tab === 'tracking' && <ScheduleTimeline stops={vehicle.stops} nextStop={vehicle.nextStop} now={gpsTime}
      stopTime={stop => Date.parse(stop.time) / 1000} clock={moscowTime}
      momentLabel={moscowTime(vehicle.gpsEventTime)} passages={vehicle.stopEvents}
      targetStopId={vehicle.forecastStopId} live />}
    {tab === 'analytics' && <div className="detail-tab-content" role="tabpanel">
      <div className="detail-section"><h3>Маршрут и прогноз</h3><div className="metric-line"><span>Целевая остановка</span><strong>{vehicle.forecastStop ? stopName(vehicle.forecastStop.name) : '—'}</strong></div><div className="metric-line"><span>Плановое прибытие</span><strong>{moscowTime(vehicle.forecastTime)}</strong></div><div className="metric-line"><span>Прогноз прибытия</span><strong>{hasMlForecast ? moscowTime(vehicle.predictedArrivalAt ?? null) : '—'}</strong></div><div className="metric-line"><span>Прогноз задержки ML</span><strong>{minutes(hasMlForecast ? vehicle.estimateSeconds : null)}</strong></div><div className="metric-line"><span>Риск опоздания более 2 минут</span><strong>{hasMlForecast ? `${Math.round(vehicle.probability! * 100)}%` : '—'}</strong></div></div>
      <div className="detail-section"><h3>Показатели движения</h3><div className="metric-line"><span>Текущее отклонение</span><strong>{minutes(vehicle.currentDeviationSeconds)}</strong></div><div className="metric-line"><span>Средняя скорость за 5 минут</span><strong>{vehicle.meanSpeed5m === null ? '—' : `${Math.round(vehicle.meanSpeed5m)} км/ч`}</strong></div><div className="metric-line"><span>Простой</span><strong>{vehicle.stoppedDurationSeconds === null ? '—' : `${Math.round(vehicle.stoppedDurationSeconds)} с`}</strong></div></div>
      {!!vehicle.stopEvents?.length && <div className="detail-section"><h3>Проходы остановок по GPS</h3>{vehicle.stopEvents.slice(-5).reverse().map((event, index) => <div className="metric-line" key={`${event.stopId}-${index}`}><span>{stopName(event.name)}</span><strong>{moscowTime(event.time)} · {minutes(event.delaySeconds)}</strong></div>)}</div>}
    </div>}
    {tab === 'details' && <div className="detail-tab-content" role="tabpanel"><div className="detail-section"><h3>Телеметрия NDTP</h3><div className="metric-line"><span>Время GPS в записи</span><strong>{vehicle.gpsEventTime ? new Date(vehicle.gpsEventTime).toLocaleString('ru-RU', { timeZone: 'Europe/Moscow' }) : '—'}</strong></div><div className="metric-line"><span>Пакет получен</span><strong>{Math.round(vehicle.gpsAgeMin * 60)} с назад</strong></div><div className="metric-line"><span>Координаты</span><strong>{vehicle.position[0].toFixed(5)}, {vehicle.position[1].toFixed(5)}</strong></div><div className="metric-line"><span>Положение на карте</span><strong>{vehicle.positionMethod === 'route' ? 'Оценка по маршруту' : vehicle.positionMethod === 'heading' ? 'Оценка по курсу' : 'Последний GPS'}</strong></div><div className="metric-line"><span>Устройство NDTP</span><strong>{vehicle.id}</strong></div><div className="metric-line"><span>Рейс</span><strong>{vehicle.tripId ?? '—'}</strong></div></div><p className="live-note">Время GPS и расписания взято из записи; свежесть на экране считается от фактического приёма пакета.</p></div>}
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
      }}><span className="route-group-icon"><Route size={19} /></span><span className="route-group-text"><strong>{route.title}</strong><small>{route.buses.length} ТС в выбранный момент</small></span><ChevronDown size={17} /></button>
      {expanded === route.id && <div className="route-buses">{route.buses.map(vehicle => <button key={vehicle.id} className="route-bus-button" onClick={() => select(vehicle.id)}><BusFront size={18} /><span><strong>ТС {vehicle.id}</strong><small>{vehicle.nextStop ? `Следующая: ${stopName(vehicle.nextStop.name)}` : 'Остановка не указана'}</small></span><span className={`status-badge ${statusOf(vehicle)}`}>{statusLabelOf(vehicle)}</span><ChevronRight size={16} /></button>)}</div>}
    </div>) : <div className="empty-state">В выбранный момент транспорта нет</div>}</div>
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
  return <div className="data-view"><div className="data-intro"><h2>{title}</h2><p>{view === 'events' ? 'Отклонения от расписания в выбранном срезе' : 'Текущие координаты и состояние транспорта'}</p></div><div className="list-grid">{shown.length ? shown.map(v => <button className="grid-row" key={v.id} onClick={() => select(v.id)}><span className={`row-icon ${statusOf(v)}`}><BusFront size={21} /></span><span><strong>ТС {v.id}</strong><small>{v.nextStop ? stopName(v.nextStop.name) : 'Без остановки'}</small></span><span className={`status-badge ${statusOf(v)}`}>{statusLabelOf(v)}</span><ChevronRight size={18} /></button>) : <div className="empty-state">В этом срезе событий нет</div>}</div></div>
}

const navItems: { id: View; label: string; icon: React.ElementType }[] = [
  { id: 'map', label: 'Карта', icon: MapIcon },
  { id: 'vehicles', label: 'Транспорт', icon: BusFront },
  { id: 'events', label: 'События', icon: AlertTriangle },
  { id: 'analytics', label: 'Аналитика', icon: BarChart3 },
]

function App() {
  const [time, setTime] = useState(plan.start)
  const [playing, setPlaying] = useState(initialAutoplay)
  const [mode, setMode] = useState<Mode>(initialMode)
  const activePlan = plan
  const startTime = activePlan.start
  const endTime = activePlan.end
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const [modelStatus, setModelStatus] = useState('ready')
  const liveRouteGeometry = JSON.stringify(
    vehicles.filter(vehicle => vehicle.stops.length >= 2)
      .map(({ id, direction, stops }) =>
        [id, direction, stops.map(({ lat, lon }) => [lat, lon])])
      .sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
  )
  const liveNetwork = useMemo<SharedNetwork>(() => {
    const segments: [number, number][][] = []
    const owners: Record<string, number[]> = {}
    for (const vehicle of vehicles) {
      if (vehicle.stops.length < 2) continue
      const index = segments.push(vehicle.stops.map(stop => [stop.lat, stop.lon])) - 1
      owners[vehicle.id] = [index]
      if (vehicle.direction) owners[`${vehicle.id}:${vehicle.direction}`] = [index]
    }
    return { mergeMeters: 35, nodes: [], edges: [],
      roadRoutes: { segments, vehicles: owners } }
  }, [liveRouteGeometry])
  const network = mode === 'live' ? liveNetwork : historicalNetwork
  const timeRef = useRef(time)
  const liveMatches = useRef(new Map<string, { gpsEventTime: string; routeKey: string;
    raw: [number, number]; match: RoadMatch }>())
  timeRef.current = time
  const [speed, setSpeed] = useState(60)
  const [monitorCollapsed, setMonitorCollapsed] = useState(false)
  const [view, setView] = useState<View>('map')
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null)
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
        const url = mode === 'live' ? '/api/live/snapshot'
          : `/api/${mode}/snapshot?at=${requestedTime}${mode === 'replay' && speed === 1 ? '&on_demand=true' : ''}`
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
  }, [mode, speed])
  const displayVehicles = mode === 'live' ? vehicles.map(vehicle => {
    const prior = liveMatches.current.get(vehicle.id)
    const routeKey = vehicle.direction ? `${vehicle.id}:${vehicle.direction}` : prior?.routeKey ?? vehicle.id
    const eventTime = vehicle.gpsEventTime ?? ''
    if (eventTime && (prior?.gpsEventTime !== eventTime || prior.routeKey !== routeKey)) {
      const ageSeconds = prior && prior.routeKey === routeKey
        ? (Date.parse(eventTime) - Date.parse(prior.gpsEventTime)) / 1000 : Infinity
      const rawMove = prior ? Math.hypot(
        (vehicle.position[0] - prior.raw[0]) * 111_320,
        (vehicle.position[1] - prior.raw[1]) * 111_320 * Math.cos(vehicle.position[0] * Math.PI / 180)) : 0
      const previousProgress = ageSeconds > 0 && ageSeconds <= 180 ? prior?.match.progressMeters : undefined
      const match = matchRoadFix(vehicle.position, vehicle.heading, vehicle.speed, network, routeKey,
        previousProgress, Math.max(60, rawMove * 2 + 40),
        previousProgress === undefined ? undefined : prior?.match.position)
      if (match) {
        liveMatches.current.set(vehicle.id, { gpsEventTime: eventTime, routeKey,
          raw: vehicle.position, match })
      }
    }
    const current = liveMatches.current.get(vehicle.id)
    if (current && current.routeKey === routeKey) {
      const progress = current.match.progressMeters + Math.max(0, vehicle.speed) *
        Math.min(30, Math.max(0, vehicle.gpsAgeMin * 60)) / 3.6
      const position = pointOnRoadRoute(network, routeKey, progress) ?? current.match.position
      return { ...vehicle, position, positionMethod: 'route' as const,
        routeKey, routeProgress: progress }
    }
    const estimated = estimateBetweenFixes(vehicle.position, vehicle.speed, vehicle.heading,
      Math.max(0, vehicle.gpsAgeMin * 60), network, routeKey)
    return { ...vehicle, position: estimated.position, heading: estimated.heading ?? vehicle.heading,
      positionMethod: estimated.method, routeKey }
  }) : vehicles.flatMap(vehicle => {
    if (!vehicle.gpsEventTime) return [vehicle]
    const fixTime = stopSeconds(vehicle.gpsEventTime, activePlan.date)
    if (fixTime > time) return []
    const estimated = estimateBetweenFixes(vehicle.position, vehicle.speed, vehicle.heading,
      time - fixTime, network, vehicle.id)
    return [{ ...vehicle, position: estimated.position, heading: estimated.heading ?? vehicle.heading,
      positionMethod: estimated.method,
      gpsAgeMin: (time - fixTime) / 60 }]
  })
  const selected = displayVehicles.find(v => v.id === selectedId) ?? null
  const filtered = displayVehicles.filter(v => {
    const status = statusOf(v)
    if (filter === 'normal' && status !== 'normal') return false
    const deviation = mode === 'live' ? v.currentDeviationSeconds : v.estimateSeconds
    if (filter === 'deviation' && (deviation === null || Math.abs(deviation) < 60)) return false
    if (filter === 'risk' && !['minor', 'critical'].includes(status)) return false
    return `${v.id} ${v.nextStop?.name ?? ''}`.toLocaleLowerCase('ru').includes(search.toLocaleLowerCase('ru'))
  })
  const countRisk = displayVehicles.filter(v => statusOf(v) === 'critical').length
  const switchMode = (next: Mode) => {
    liveMatches.current.clear()
    const url = new URL(window.location.href)
    url.searchParams.delete('autoplay')
    url.searchParams.delete('demo')
    if (next === 'live') url.searchParams.set('mode', 'live')
    else url.searchParams.delete('mode')
    window.history.replaceState(null, '', url)
    setMode(next)
    setPlaying(false)
    setTime(plan.start)
    setSelectedId(null)
    setSelectedRouteId(null)
    setVehicles([])
    setSpeed(60)
  }
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
      <header className="topbar"><div className="topbar-title"><button className="mobile-menu" aria-label="Открыть меню" onClick={() => setMobileNav(true)}><Menu size={22} /></button><h1>{view === 'map' ? 'Карта' : navItems.find(item => item.id === view)?.label}</h1></div><div className="topbar-actions"><div className="mode-picker" aria-label="Источник данных"><button className={mode === 'replay' ? 'active' : ''} onClick={() => switchMode('replay')}>История</button><button className={mode === 'live' ? 'active' : ''} onClick={() => switchMode('live')}>NDTP</button></div><div className="date-chip"><CalendarDays size={16} /> {mode === 'live' ? 'Прямой эфир' : activePlan.date}</div><div className="time-select"><Clock3 size={17} />{mode === 'live' ? 'сейчас' : `${formatTime(time, true)} МСК`}</div></div></header>
      {apiError && <div className="service-banner">Нет связи с backend: {apiError}. Показано последнее полученное состояние.</div>}
      {modelStatus === 'unavailable' && !apiError && <div className="service-banner">ML-сервис недоступен: положение ТС обновляется, новые прогнозы временно не рассчитываются.</div>}
      {mode !== 'live' && <div className="timeline-bar"><button className="play-button" onClick={() => { if (time >= endTime) setTime(startTime); setPlaying(!playing || time >= endTime) }} aria-label={playing ? 'Пауза' : 'Воспроизвести'} title={playing ? 'Пауза' : 'Воспроизвести'}>{playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}</button><span className="timeline-time">{formatTime(time, true)}</span><input className="time-slider" type="range" min={startTime} max={endTime} step="1" value={time} onChange={event => { setPlaying(false); setTime(Number(event.target.value)) }} aria-valuetext={`${formatTime(time, true)} МСК`} aria-label="Выбрать время в датасете" style={{ background: `linear-gradient(to right, #1766ef ${(time - startTime) / (endTime - startTime) * 100}%, #dce5f2 0)` }} /><span className="timeline-end">{formatTime(endTime)}</span><label className="speed-control">Скорость <select value={speed} onChange={event => setSpeed(Number(event.target.value))} aria-label="Скорость воспроизведения">{playbackSpeeds.map(value => <option key={value} value={value}>{value}×</option>)}</select><ChevronDown size={13} /></label></div>}
      <div className="workspace">
        <section className="primary-panel">
          {view === 'map' ? <TransportMap vehicles={displayVehicles} selected={selected} selectedId={selectedRouteId ?? selectedId} onSelect={select} onRouteSelect={mode === 'live' ? select : selectRoute} showRoutes={showRoutes} setShowRoutes={setShowRoutes} collapsed={monitorCollapsed} network={network} replayTime={mode === 'live' ? null : time} /> : <DataView view={view} vehicles={displayVehicles} select={select} routes={mode === 'live' ? [] : historicalRoutes} selectRoute={selectRouteInList} />}
        </section>
        {monitorCollapsed && <button className="restore-monitor" onClick={() => setMonitorCollapsed(false)} title="Открыть мониторинг" aria-label="Открыть мониторинг"><PanelRightOpen size={19} /></button>}
        {monitorCollapsed ? null : <aside className="right-panel">
          {selectedRouteId ? <>
            <button className="collapse-detail" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг" aria-label="Свернуть мониторинг"><PanelRightClose size={18} /></button>
            <RouteDetail id={selectedRouteId} stops={activePlan.vehicles.find(vehicle => vehicle.id === selectedRouteId)?.stops ?? []} onBack={() => setSelectedRouteId(null)} />
          </> : selectedId ? <>
            <button className="collapse-detail" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг" aria-label="Свернуть мониторинг"><PanelRightClose size={18} /></button>
            {selected ? mode === 'live' ? <LiveDetail vehicle={selected} onBack={() => setSelectedId(null)} /> : <VehicleDetail vehicle={selected} time={time} day={activePlan.date} onBack={() => setSelectedId(null)} onJumpToForecast={pointTime => { setPlaying(false); setTime(pointTime) }} /> : <InactiveRouteDetail id={selectedId} onBack={() => setSelectedId(null)} />}
          </> : <>
            <div className="panel-header">
              <button className="monitor-heading" onClick={() => setMonitorCollapsed(true)} title="Свернуть мониторинг"><span className="eyebrow">МОНИТОРИНГ</span><span className="monitor-title">Транспорт на линии <span>{displayVehicles.length}</span></span></button>
              <button className="panel-icon-button" title="Свернуть мониторинг" aria-label="Свернуть мониторинг" onClick={() => setMonitorCollapsed(true)}><PanelRightClose size={19} /></button>
            </div>
            <div className="search-box"><Search size={18} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Номер ТС или остановка" aria-label="Поиск транспорта" />{search && <button aria-label="Очистить поиск" onClick={() => setSearch('')}><X size={16} /></button>}</div>
            <div className="filter-row"><button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>Все</button><button className={filter === 'deviation' ? 'active' : ''} onClick={() => setFilter('deviation')}>Отклонения</button><button className={filter === 'risk' ? 'active' : ''} onClick={() => setFilter('risk')}>Риск</button></div>
            <div className="vehicle-list">{filtered.length ? filtered.map(vehicle => <VehicleRow key={vehicle.id} vehicle={vehicle} active={false} live={mode === 'live'} onClick={() => setSelectedId(vehicle.id)} />) : <div className="empty-state">Транспорт не найден</div>}</div>
            <div className="panel-footer"><span className="footer-dot" /> {filtered.length} из {displayVehicles.length} ТС · {mode === 'live' ? 'прямой эфир' : `${formatTime(time)} МСК`}</div>
          </>}
        </aside>}
      </div>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
