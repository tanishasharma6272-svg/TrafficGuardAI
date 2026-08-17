/**
 * Aggregate location counts grouped by risk level.
 * @param {Array} risks - List of risk objects
 * @returns {Object} Count mapping { Critical: 0, High: 0, Medium: 0, Low: 0 }
 */
export function countByLevel(risks) {
  return risks.reduce(
    (summary, item) => {
      if (item.risk_level in summary) {
        summary[item.risk_level] = summary[item.risk_level] + 1
      }
      return summary
    },
    { Critical: 0, High: 0, Medium: 0, Low: 0 },
  )
}

/**
 * Calculate total police officers stationed across all monitored locations.
 * @param {Array} risks - List of risk objects
 * @returns {number} Total count
 */
export function totalOfficers(risks) {
  return risks.reduce((total, item) => total + (item.police_officers || 0), 0)
}

/**
 * Compute the derived Average Network Risk score across all locations.
 * NOTE: This is a frontend-derived arithmetic mean of current /risk scores,
 * not a backend-defined composite network risk model.
 * @param {Array} risks - List of risk objects
 * @returns {number} Average score rounded to 1 decimal place
 */
export function calculateAverageRisk(risks) {
  if (!risks || risks.length === 0) return 0
  const sum = risks.reduce((acc, item) => acc + (item.risk_score || 0), 0)
  return Number((sum / risks.length).toFixed(1))
}

/**
 * Build a lookup map of location ID to Location object.
 * @param {Array} locations - List of location objects
 * @returns {Map<number, Object>}
 */
export function buildLocationMap(locations) {
  const map = new Map()
  locations.forEach((location) => map.set(location.id, location))
  return map
}

/**
 * Filter and sort locations that have Critical or High risk severity.
 * @param {Array} risks - List of risk objects
 * @returns {Array} High-risk locations sorted by score descending
 */
export function getHighRiskLocations(risks) {
  return risks
    .filter((item) => item.risk_level === 'Critical' || item.risk_level === 'High')
    .sort((a, b) => b.risk_score - a.risk_score)
}

/**
 * Helper to format geographic coordinate pairs for display.
 * @param {number} lat - Latitude
 * @param {number} lng - Longitude
 * @returns {string} Formatted string
 */
export function formatCoordinates(lat, lng) {
  if (typeof lat !== 'number' || typeof lng !== 'number') return '—'
  return `${lat.toFixed(4)}, ${lng.toFixed(4)}`
}
