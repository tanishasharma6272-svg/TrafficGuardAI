import { LEVEL_COLOR } from '../../constants/risk'

export function RiskGauge({ score, level }) {
  const safeScore = typeof score === 'number' ? score : 0
  const pct = Math.max(0, Math.min(100, safeScore))
  const color = LEVEL_COLOR[level] || '#8c84a8'

  return (
    <div
      className="risk-gauge"
      style={{ '--pct': pct, '--gauge-color': color }}
      role="img"
      aria-label={`Risk score: ${safeScore.toFixed(1)} out of 100 (${level})`}
    >
      <div className="risk-gauge-inner">
        <span className="gauge-score mono">{safeScore.toFixed(0)}</span>
        <span className="gauge-max">/ 100</span>
        <span className="gauge-level-tag">{level}</span>
      </div>
    </div>
  )
}
