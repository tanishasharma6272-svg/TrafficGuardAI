export function Badge({ children, variant = 'demo', dot = false, pulse = false }) {
  return (
    <span className={`tg-badge tg-badge--${variant}`}>
      {dot && <span className={`tg-badge-dot ${pulse ? 'tg-badge-dot--pulse' : ''}`} />}
      {children}
    </span>
  )
}
