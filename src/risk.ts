export type RiskStatus = 'unknown' | 'normal' | 'minor' | 'critical'

export function riskStatus(probability: number | null): RiskStatus {
  if (probability === null || !Number.isFinite(probability)) return 'unknown'
  if (probability >= 0.9) return 'critical'
  if (probability >= 0.7) return 'minor'
  return 'normal'
}
