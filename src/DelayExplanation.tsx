export type DelayFactor = { feature: string; impact_seconds: number; value: string | number | null }
export type DelayExplanationData = {
  base_seconds: number
  current_deviation_seconds: number
  other_seconds: number
  factors: DelayFactor[]
}

const labels: Record<string, [string, string]> = {
  tr_id: ['Номер транспортного средства', ''],
  cur_dev_s: ['Текущее отклонение', 'с'],
  previous_cur_dev_s: ['Отклонение при предыдущем расчёте', 'с'],
  previous_delay_age_s: ['Время с прошлого расчёта', 'с'],
  previous_delay_available: ['Есть предыдущий расчёт', ''],
  delay_change_since_previous_s: ['Изменение задержки с прошлого расчёта', 'с'],
  delay_change_per_min: ['Темп изменения задержки', 'с/мин'],
  cur_dev_lag_2: ['Отклонение два расчёта назад', 'с'],
  lag_2_age_s: ['Время с предпоследнего расчёта', 'с'],
  lag_2_available: ['Есть предпоследний расчёт', ''],
  delay_change_lag_2_s: ['Изменение задержки за два расчёта', 'с'],
  delay_change_lag_2_per_min: ['Темп изменения за два расчёта', 'с/мин'],
  realtime_data_age_s: ['Возраст последней телеметрии', 'с'],
  realtime_speed_age_s: ['Возраст последней скорости', 'с'],
  realtime_gps_age_s: ['Возраст последнего GPS', 'с'],
  realtime_speed_last: ['Последняя скорость', 'км/ч'],
  realtime_speed_std_2m: ['Разброс скорости за 2 мин', 'км/ч'],
  realtime_speed_std_5m: ['Разброс скорости за 5 мин', 'км/ч'],
  realtime_speed_std_10m: ['Разброс скорости за 10 мин', 'км/ч'],
  realtime_packet_count_2m: ['GPS-пакетов за 2 мин', ''],
  realtime_packet_count_5m: ['GPS-пакетов за 5 мин', ''],
  realtime_packet_count_10m: ['GPS-пакетов за 10 мин', ''],
  realtime_speed_count_2m: ['Замеров скорости за 2 мин', ''],
  realtime_speed_count_5m: ['Замеров скорости за 5 мин', ''],
  realtime_speed_count_10m: ['Замеров скорости за 10 мин', ''],
  realtime_speed_mean_2m: ['Средняя скорость за 2 мин', 'км/ч'],
  realtime_speed_mean_5m: ['Средняя скорость за 5 мин', 'км/ч'],
  realtime_speed_mean_10m: ['Средняя скорость за 10 мин', 'км/ч'],
  realtime_stopped_fraction_2m: ['Доля остановок за 2 мин', ''],
  realtime_stopped_fraction_5m: ['Доля остановок за 5 мин', ''],
  realtime_stopped_fraction_10m: ['Доля остановок за 10 мин', ''],
  realtime_distance_to_target_m: ['Расстояние до цели', 'м'],
  context_vehicle_count_500m: ['Соседних ТС в 500 м', ''],
  context_speed_count_500m: ['Скоростей соседних ТС в 500 м', ''],
  context_speed_mean_500m: ['Скорость соседних ТС', 'км/ч'],
  context_speed_median_500m: ['Медианная скорость соседних ТС', 'км/ч'],
  context_stopped_fraction_500m: ['Доля стоящих соседних ТС', ''],
  schedule_horizon_s: ['До планового прибытия', 'с'],
  schedule_target_lon: ['Долгота целевой остановки', '°'],
  schedule_target_lat: ['Широта целевой остановки', '°'],
  schedule_target_address: ['Адрес целевой остановки', ''],
  schedule_stops_remaining: ['Осталось остановок', ''],
  route_signature: ['Маршрут', ''],
  section_id: ['Участок маршрута', ''],
  section_duration_s: ['Плановое время участка', 'с'],
  section_planned_progress: ['Доля пройденного участка по плану', ''],
  planned_section_available: ['Есть план участка', ''],
  stop_lon: ['Долгота текущей остановки', '°'],
  stop_lat: ['Широта текущей остановки', '°'],
  next_lon: ['Долгота следующей остановки', '°'],
  next_lat: ['Широта следующей остановки', '°'],
  historical_hour: ['Время суток', 'ч'],
  historical_day_of_week: ['День недели', ''],
  historical_time_sin: ['Время суток: циклическая компонента 1', ''],
  historical_time_cos: ['Время суток: циклическая компонента 2', ''],
}

function valueOf(factor: DelayFactor) {
  if (factor.value === null) return 'нет данных'
  const [, unit] = labels[factor.feature] ?? [factor.feature, '']
  if (typeof factor.value !== 'number') return factor.value.length > 28 ? `${factor.value.slice(0, 28)}…` : factor.value
  const value = factor.feature.includes('fraction') || factor.feature === 'section_planned_progress'
    ? `${Math.round(factor.value * 100)}%`
    : factor.feature.endsWith('_lon') || factor.feature.endsWith('_lat')
      ? factor.value.toFixed(4)
      : Number(factor.value.toFixed(1)).toString()
  return `${value}${unit ? ` ${unit}` : ''}`
}

export default function DelayExplanation({ data }: { data: DelayExplanationData }) {
  return <div className="shap-explanation">
    <strong>Почему модель дала такой прогноз</strong>
    <p>TreeSHAP для прогноза отклонения в секундах. Вклад «+» увеличивает прогноз, «−» уменьшает.</p>
    <div className="shap-factor"><span>Уже накоплено по графику</span><b>{Math.round(data.current_deviation_seconds)} с</b></div>
    <div className="shap-factor"><span>Базовый прогноз изменения задержки</span><b>{Math.round(data.base_seconds)} с</b></div>
    {data.factors.map(factor => <div className="shap-factor" key={factor.feature}>
      <span>{labels[factor.feature]?.[0] ?? factor.feature}<small>{valueOf(factor)}</small></span>
      <b className={factor.impact_seconds >= 0 ? 'up' : 'down'}>{factor.impact_seconds >= 0 ? '+' : '−'}{Math.round(Math.abs(factor.impact_seconds))} с</b>
    </div>)}
    <div className="shap-factor"><span>Остальные признаки</span><b>{data.other_seconds >= 0 ? '+' : '−'}{Math.round(Math.abs(data.other_seconds))} с</b></div>
    <p className="what-if-note">Показаны четыре крупнейших вклада. TreeSHAP объясняет решение модели относительно её среднего ответа; это признаки, связанные с прогнозом, а не доказанные причины задержки.</p>
  </div>
}
