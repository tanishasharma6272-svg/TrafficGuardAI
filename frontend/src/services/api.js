import axios from 'axios'

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
})

// ---------------------------------------------------------------------------
// PostgreSQL Monitored Locations & Legacy Endpoints
// ---------------------------------------------------------------------------

/**
 * Fetch all simulated traffic monitoring locations.
 * @returns {Promise<Array>} List of location objects
 */
export async function fetchLocations() {
  const response = await api.get('/locations')
  return response.data
}

/**
 * Fetch summarized risk assessments for all locations (Legacy Rule Engine).
 * @returns {Promise<Array>} List of risk summary objects
 */
export async function fetchRiskOverview() {
  const response = await api.get('/risk')
  return response.data
}

/**
 * Fetch detailed risk calculation breakdown for a specific location (Legacy Rule Engine).
 * @param {number|string} locationId - The unique ID of the location
 * @returns {Promise<Object>} Detailed risk object with contributing factors
 */
export async function fetchLocationRisk(locationId) {
  const response = await api.get(`/risk/${locationId}`)
  return response.data
}

// ---------------------------------------------------------------------------
// Dedicated ML Risk & SHAP Explainability Endpoints
// ---------------------------------------------------------------------------

/**
 * Fetch ML-predicted risk overview for all monitored locations.
 * @returns {Promise<Array>} List of ML risk summaries
 */
export async function fetchMLRiskOverview() {
  const response = await api.get('/api/ml/risk')
  return response.data
}

/**
 * Fetch comprehensive ML risk breakdown for a single location by ID.
 * @param {number|string} locationId - Unique location identifier
 * @returns {Promise<Object>} Detailed ML risk detail object
 */
export async function fetchMLLocationRisk(locationId) {
  const response = await api.get(`/api/ml/risk/${locationId}`)
  return response.data
}

/**
 * Fetch SHAP feature attribution vectors and model explainability for a location.
 * @param {number|string} locationId - Unique location identifier
 * @returns {Promise<Object>} SHAP explanation object with positive and negative contributors
 */
export async function fetchRiskExplanation(locationId) {
  const response = await api.get(`/api/ml/explain/${locationId}`)
  return response.data
}

// ---------------------------------------------------------------------------
// Police Deployment Optimizer Endpoints
// ---------------------------------------------------------------------------

/**
 * Request optimal police unit placement from the greedy deployment optimizer.
 * @param {Object} params - { available_units: number, coverage_radius_km: number, min_risk_level?: string }
 * @returns {Promise<Object>} Deployment recommendation payload with selected units and metrics
 */
export async function recommendDeployment(params) {
  const response = await api.post('/api/deployment/recommend', {
    available_units: Number(params.available_units),
    coverage_radius_km: Number(params.coverage_radius_km),
    ...(params.min_risk_level ? { min_risk_level: params.min_risk_level } : {}),
  })
  return response.data
}
