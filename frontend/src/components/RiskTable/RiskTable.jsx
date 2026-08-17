import { RISK_LEVELS } from '../../constants/risk'
import { RiskPill } from '../shared/RiskPill'
import { SortableHeader } from './SortableHeader'
import { formatCoordinates } from '../../utils/riskStats'

export function RiskTable({
  sortedRisks,
  totalCount,
  selectedId,
  onSelectLocation,
  query,
  setQuery,
  activeLevel,
  setActiveLevel,
  sortKey,
  sortDir,
  onSort,
  onClearFilters,
  hasActiveFilters,
}) {
  return (
    <section className="table-panel" aria-label="Location Risk Overview Table">
      <div className="panel-header">
        <div>
          <div className="panel-title-with-badge">
            <h2 className="panel-title">Location Risk Overview</h2>
            <span className="count-pill">
              Showing {sortedRisks.length} of {totalCount} locations
            </span>
          </div>
          <p className="panel-subtitle">Comprehensive tabular registry synchronized with map &amp; details</p>
        </div>

        <div className="table-toolbar">
          <div className="search-input-wrapper">
            <svg
              className="search-icon"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="7" />
              <path d="M21 21l-4.3-4.3" strokeLinecap="round" />
            </svg>
            <input
              type="text"
              className="search-input"
              placeholder="Search intersection…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Filter locations by name"
            />
            {query && (
              <button
                type="button"
                className="search-clear-btn"
                onClick={() => setQuery('')}
                aria-label="Clear search input"
              >
                ×
              </button>
            )}
          </div>

          <div className="chip-filter-row" role="group" aria-label="Filter by risk severity">
            {['All', ...RISK_LEVELS].map((level) => {
              const isActive = activeLevel === level
              const tone = level !== 'All' ? level.toLowerCase() : ''
              return (
                <button
                  key={level}
                  type="button"
                  className={`filter-chip ${isActive ? 'is-active' : ''} ${
                    tone ? `filter-chip--${tone}` : ''
                  }`}
                  onClick={() => setActiveLevel(level)}
                >
                  {level}
                </button>
              )
            })}

            {hasActiveFilters && (
              <button
                type="button"
                className="clear-all-chip"
                onClick={onClearFilters}
                title="Reset all active search & filters"
              >
                Reset
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="table-scroll-container">
        <table className="operational-table">
          <thead>
            <tr>
              <SortableHeader
                label="Location"
                sortKey="name"
                activeKey={sortKey}
                dir={sortDir}
                onSort={onSort}
              />
              <SortableHeader
                label="Risk Severity"
                sortKey="level"
                activeKey={sortKey}
                dir={sortDir}
                onSort={onSort}
              />
              <SortableHeader
                label="Score (0-100)"
                sortKey="score"
                activeKey={sortKey}
                dir={sortDir}
                onSort={onSort}
                className="th-mono"
              />
              <SortableHeader
                label="Police Units"
                sortKey="police"
                activeKey={sortKey}
                dir={sortDir}
                onSort={onSort}
                className="th-mono"
              />
              <SortableHeader
                label="Speed / Free Flow"
                sortKey="speed"
                activeKey={sortKey}
                dir={sortDir}
                onSort={onSort}
                className="th-mono"
              />
              <th className="th-mono">Coordinates (GPS)</th>
            </tr>
          </thead>
          <tbody>
            {sortedRisks.map(({ risk, location }) => {
              const isSelected = risk.id === selectedId
              return (
                <tr
                  key={risk.id}
                  className={`table-row ${isSelected ? 'is-selected' : ''}`}
                  onClick={() => onSelectLocation(risk.id)}
                  role="row"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onSelectLocation(risk.id)
                    }
                  }}
                  aria-selected={isSelected}
                >
                  <td className="td-name">
                    <button
                      type="button"
                      className="location-row-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectLocation(risk.id)
                      }}
                    >
                      {risk.name}
                    </button>
                  </td>
                  <td>
                    <RiskPill level={risk.risk_level} size="small" />
                  </td>
                  <td className="mono td-score">{risk.risk_score.toFixed(2)}</td>
                  <td className="mono td-police">{risk.police_officers} units</td>
                  <td className="mono td-speed">
                    {location
                      ? `${location.traffic_speed} / ${location.free_flow_speed} km/h`
                      : '—'}
                  </td>
                  <td className="mono td-coords">
                    {location
                      ? formatCoordinates(location.latitude, location.longitude)
                      : '—'}
                  </td>
                </tr>
              )
            })}

            {sortedRisks.length === 0 && (
              <tr>
                <td colSpan={6} className="empty-table-cell">
                  <div className="empty-table-state">
                    <p>No locations match the current search or risk filter.</p>
                    {hasActiveFilters && (
                      <button
                        type="button"
                        className="reset-search-btn"
                        onClick={onClearFilters}
                      >
                        Clear Active Filters
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}
