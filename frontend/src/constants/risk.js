export const RISK_LEVELS = ['Critical', 'High', 'Medium', 'Low']

export const LEVEL_COLOR = {
  Critical: '#F98CA3',
  High: '#FFB88C',
  Medium: '#FFDE8C',
  Low: '#95DEC0',
}

export const LEVEL_ACCENT = {
  Critical: '#E06784',
  High: '#E2914F',
  Medium: '#CF9F2E',
  Low: '#4C9C7C',
}

export const LEVEL_BG = {
  Critical: 'rgba(249, 140, 163, 0.18)',
  High: 'rgba(255, 184, 140, 0.18)',
  Medium: 'rgba(255, 222, 140, 0.22)',
  Low: 'rgba(149, 222, 192, 0.22)',
}

// Nagpur bounding box, derived from the monitored network
export const NAGPUR_CENTER = [21.1458, 79.0882]
export const NAGPUR_BOUNDS = [
  [21.0922, 79.0325],
  [21.1857, 79.136],
]

// Data Status Modes
export const DATA_MODES = {
  DEMO: 'DEMO DATA',
  LIVE: 'LIVE SENSOR',
  STALE: 'STALE DATA',
  ERROR: 'CONNECTION ERROR',
}

// System Status Modes
export const SYSTEM_STATUS = {
  OPERATIONAL: 'OPERATIONAL',
  DEGRADED: 'DEGRADED',
  OFFLINE: 'OFFLINE',
}
