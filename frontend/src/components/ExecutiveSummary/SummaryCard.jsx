export function SummaryCard({
  label,
  value,
  tone,
  sublabel,
  isActive = false,
  onClick,
  isInteractive = true,
}) {
  return (
    <div
      className={`summary-card summary-card--${tone} ${isActive ? 'is-active' : ''} ${
        isInteractive && onClick ? 'is-clickable' : ''
      }`}
      onClick={onClick}
      role={onClick ? 'button' : 'article'}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onClick()
              }
            }
          : undefined
      }
      aria-label={`${label}: ${value}`}
    >
      <div className="summary-card-header">
        <span className="summary-card-label">{label}</span>
        {isActive && <span className="summary-active-pill">FILTERED</span>}
      </div>

      <strong className="summary-card-val mono">{value}</strong>

      {sublabel && <span className="summary-card-sublabel">{sublabel}</span>}
    </div>
  )
}
