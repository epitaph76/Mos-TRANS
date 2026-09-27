import { useState } from 'react'
import { scheduledCycleMinutes, serviceScenario } from './whatIfModel'

type Stop = { time: string; name: string }

export default function WhatIf({ vehicleId, stops }: { vehicleId: string; stops: Stop[] }) {
  const plannedCycle = scheduledCycleMinutes(stops)
  const [current, setCurrent] = useState(1)
  const [additional, setAdditional] = useState(0)
  const [cycle, setCycle] = useState(plannedCycle ?? 90)
  const scenario = serviceScenario(cycle, current, additional)
  return <section className="detail-section what-if" aria-label="Сценарий дополнительных автобусов">
    <h3>Что если добавить автобусы?</h3>
    <p>Маршрут из расписания ТС {vehicleId}. {plannedCycle === null
      ? 'Полный оборот маршрута неизвестен: введите его вручную.'
      : `Типичный полный оборот по возвращению к начальной остановке: ${plannedCycle} мин.`}</p>
    <div className="what-if-controls"><button className="what-if-add" onClick={() => setAdditional(value => Math.min(100, value + 1))}>+ Автобус на маршрут</button><button className="what-if-remove" onClick={() => setAdditional(value => Math.max(0, value - 1))} disabled={additional === 0} aria-label="Убрать один добавленный автобус">−</button><span>Добавлено: {additional}</span></div>
    <div className="what-if-inputs">
      <label>Автобусов сейчас<input type="number" min="1" max="200" value={current} onChange={event => setCurrent(Math.max(1, Math.min(200, Number(event.target.value) || 1)))} /></label>
      <label>Полный оборот, мин<input type="number" min="10" max="600" value={cycle} onChange={event => setCycle(Math.max(10, Math.min(600, Number(event.target.value) || 10)))} /></label>
    </div>
    <p className="what-if-note">В выгрузке подтверждён один автобус этой линии. Если на маршруте их больше, укажите фактический выпуск выше.</p>
    <div className="metric-line"><span>Интервал между автобусами</span><strong>{scenario.beforeHeadway.toFixed(1)} → {scenario.afterHeadway.toFixed(1)} мин</strong></div>
    <div className="metric-line"><span>Рейсов в час через остановку</span><strong>{scenario.beforeDepartures.toFixed(1)} → {scenario.afterDepartures.toFixed(1)}</strong></div>
    <div className="metric-line"><span>Среднее ожидание при равномерном выпуске</span><strong>{scenario.beforeWait.toFixed(1)} → {scenario.afterWait.toFixed(1)} мин</strong></div>
    <p className="what-if-note">Расчёт предполагает равномерный выпуск: интервал = оборот / число автобусов, среднее ожидание = половина интервала. Он показывает частоту обслуживания, а не изменение пробок или задержки уже едущего ТС. В CatBoost нет признака числа автобусов на маршруте.</p>
  </section>
}
