export function RiskPill({ level, size = 'medium' }) {
  const safeLevel = level || 'Low'
  const tone = safeLevel.toLowerCase()

  return (
    <span className={`risk-pill risk-pill--${tone} risk-pill--${size}`}>
      <span className="risk-pill-dot" />
      {safeLevel}
    </span>
  )
}
