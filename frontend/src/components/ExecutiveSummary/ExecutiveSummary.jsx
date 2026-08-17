import { useMemo } from 'react'
import { SummaryCard } from './SummaryCard'
import { countByLevel, totalOfficers, calculateAverageRisk } from '../../utils/riskStats'

export function ExecutiveSummary({
  risks,
  activeLevel,
  onSelectLevel,
  onClearFilter,
  hasActiveFilter,
}) {
  const counts = useMemo(() => countByLevel(risks), [risks])
  const officers = useMemo(() => totalOfficers(risks), [risks])
  const avgRisk = useMemo(() => calculateAverageRisk(risks), [risks])

  return (
    <section className="executive-summary-section" aria-label="Executive Risk Overview">
      <div className="summary-grid">
        <SummaryCard
          label="Average Network Risk"
          value={avgRisk > 0 ? `${avgRisk}` : '—'}
          sublabel="Derived mean of current risk scores"
          tone="network"
          isInteractive={false}
        />

        <SummaryCard
          label="Critical Risk"
          value={counts.Critical}
          sublabel="Immediate attention required"
          tone="critical"
          isActive={activeLevel === 'Critical'}
          onClick={() => onSelectLevel(activeLevel === 'Critical' ? 'All' : 'Critical')}
        />

        <SummaryCard
          label="High Risk"
          value={counts.High}
          sublabel="Elevated hazard score"
          tone="high"
          isActive={activeLevel === 'High'}
          onClick={() => onSelectLevel(activeLevel === 'High' ? 'All' : 'High')}
        />

        <SummaryCard
          label="Medium Risk"
          value={counts.Medium}
          sublabel="Moderate flow friction"
          tone="medium"
          isActive={activeLevel === 'Medium'}
          onClick={() => onSelectLevel(activeLevel === 'Medium' ? 'All' : 'Medium')}
        />

        <SummaryCard
          label="Low Risk"
          value={counts.Low}
          sublabel="Nominal traffic flow"
          tone="low"
          isActive={activeLevel === 'Low'}
          onClick={() => onSelectLevel(activeLevel === 'Low' ? 'All' : 'Low')}
        />

        <SummaryCard
          label="Police Officers"
          value={officers}
          sublabel="Total deployed in dataset"
          tone="officers"
          isInteractive={false}
        />
      </div>

      {hasActiveFilter && (
        <div className="active-filter-indicator">
          <span>
            Active Filter: <strong>{activeLevel}</strong>
          </span>
          <button type="button" className="clear-filter-button" onClick={onClearFilter}>
            Clear Filter ×
          </button>
        </div>
      )}
    </section>
  )
}
