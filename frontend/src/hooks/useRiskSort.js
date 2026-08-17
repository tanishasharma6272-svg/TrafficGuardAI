import { useMemo, useState } from 'react'
import { RISK_LEVELS } from '../constants/risk'

export function useRiskSort(filteredRisks, locationById) {
  const [sortKey, setSortKey] = useState('level')
  const [sortDir, setSortDir] = useState('asc')

  const sortedRisks = useMemo(() => {
    const withLocation = filteredRisks.map((risk) => ({
      risk,
      location: locationById.get(risk.id),
    }))

    const direction = sortDir === 'asc' ? 1 : -1

    return withLocation.sort((a, b) => {
      switch (sortKey) {
        case 'name':
          return (a.risk.name || '').localeCompare(b.risk.name || '') * direction
        case 'score':
          return (a.risk.risk_score - b.risk.risk_score) * direction
        case 'police':
          return (a.risk.police_officers - b.risk.police_officers) * direction
        case 'speed': {
          const speedA = a.location?.traffic_speed ?? 0
          const speedB = b.location?.traffic_speed ?? 0
          return (speedA - speedB) * direction
        }
        case 'level':
        default: {
          const levelDiff =
            RISK_LEVELS.indexOf(a.risk.risk_level) -
            RISK_LEVELS.indexOf(b.risk.risk_level)
          if (levelDiff !== 0) return levelDiff * direction
          return (b.risk.risk_score - a.risk.risk_score) * direction
        }
      }
    })
  }, [filteredRisks, locationById, sortKey, sortDir])

  function toggleSort(key) {
    if (sortKey === key) {
      setSortDir((current) => (current === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  return { sortKey, sortDir, toggleSort, sortedRisks }
}
