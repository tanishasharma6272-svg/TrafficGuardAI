export function SortableHeader({
  label,
  sortKey,
  activeKey,
  dir,
  onSort,
  className = '',
}) {
  const isActive = sortKey === activeKey

  return (
    <th
      className={`sortable-th ${isActive ? 'is-active' : ''} ${className}`}
      onClick={() => onSort(sortKey)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onSort(sortKey)
        }
      }}
      aria-label={`Sort by ${label} in ${
        isActive && dir === 'asc' ? 'descending' : 'ascending'
      } order`}
    >
      <span className="th-content">
        {label}
        <span className="sort-arrow-icon" aria-hidden="true">
          {isActive ? (dir === 'asc' ? '▲' : '▼') : '↕'}
        </span>
      </span>
    </th>
  )
}
