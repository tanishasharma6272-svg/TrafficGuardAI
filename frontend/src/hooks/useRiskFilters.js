import { useCallback, useMemo, useState } from 'react'

export function useRiskFilters(risks) {
  const [query, setQuery] = useState('')
  const [activeLevel, setActiveLevel] = useState('All')

  const clearFilters = useCallback(() => {
    setQuery('')
    setActiveLevel('All')
  }, [])

  const hasActiveFilters = useMemo(() => {
    return query.trim() !== '' || activeLevel !== 'All'
  }, [query, activeLevel])

  const filteredRisks = useMemo(() => {
    const q = query.trim().toLowerCase()

    return risks.filter((risk) => {
      const matchesLevel =
        activeLevel === 'All' || risk.risk_level === activeLevel
      const matchesQuery =
        !q || (risk.name && risk.name.toLowerCase().includes(q))
      return matchesLevel && matchesQuery
    })
  }, [risks, query, activeLevel])

  return {
    query,
    setQuery,
    activeLevel,
    setActiveLevel,
    clearFilters,
    hasActiveFilters,
    filteredRisks,
  }
}
