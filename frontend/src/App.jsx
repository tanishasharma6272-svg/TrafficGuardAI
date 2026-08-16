import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
})

const riskOrder = ['Critical', 'High', 'Medium', 'Low']

function App() {
  const [locations, setLocations] = useState([])
  const [risks, setRisks] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadDashboard() {
      try {
        setLoading(true)
        setError('')

        const [locationsResponse, riskResponse] = await Promise.all([
          api.get('/locations'),
          api.get('/risk'),
        ])

        if (!active) return

        setLocations(locationsResponse.data)
        setRisks(riskResponse.data)

        if (riskResponse.data.length > 0) {
          setSelectedId(riskResponse.data[0].id)
        }
      } catch (requestError) {
        if (!active) return

        const message =
          requestError.response?.data?.detail ||
          requestError.message ||
          'Unable to load dashboard data.'

        setError(message)
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    loadDashboard()

    return () => {
      active = false
    }
  }, [])

  const counts = useMemo(() => {
    return risks.reduce(
      (summary, item) => {
        summary[item.risk_level] = (summary[item.risk_level] || 0) + 1
        return summary
      },
      {
        Critical: 0,
        High: 0,
        Medium: 0,
        Low: 0,
      },
    )
  }, [risks])

  const totalOfficers = useMemo(
    () => risks.reduce((total, item) => total + item.police_officers, 0),
    [risks],
  )

  const selectedRisk = risks.find((item) => item.id === selectedId)
  const selectedLocation = locations.find((item) => item.id === selectedId)

  const sortedRisks = useMemo(() => {
    return [...risks].sort((a, b) => {
      const levelDifference =
        riskOrder.indexOf(a.risk_level) - riskOrder.indexOf(b.risk_level)

      if (levelDifference !== 0) return levelDifference

      return b.risk_score - a.risk_score
    })
  }, [risks])

  if (loading) {
    return (
      <main className="app-shell">
        <div className="loading-panel">Loading TrafficGuard AI…</div>
      </main>
    )
  }

  if (error) {
    return (
      <main className="app-shell">
        <div className="error-panel">
          <h1>TrafficGuard AI</h1>
          <p>{error}</p>
          <p className="error-help">
            Make sure the FastAPI server is running at {API_BASE_URL}.
          </p>
        </div>
      </main>
    )
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand-row">
            <span className="brand-mark">TG</span>
            <h1>TrafficGuard AI</h1>
            <span className="demo-badge">DEMO DATA</span>
          </div>
          <p className="subtitle">
            Nagpur Traffic Risk &amp; Police Deployment Dashboard
          </p>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          API connected
        </div>
      </header>

      <section className="summary-grid" aria-label="Risk summary">
        <SummaryCard
          label="Critical"
          value={counts.Critical}
          tone="critical"
        />
        <SummaryCard label="High" value={counts.High} tone="high" />
        <SummaryCard label="Medium" value={counts.Medium} tone="medium" />
        <SummaryCard label="Low" value={counts.Low} tone="low" />
        <SummaryCard
          label="Police Officers"
          value={totalOfficers}
          tone="officers"
        />
      </section>

      <section className="workspace">
        <div className="map-panel">
          <div className="panel-heading">
            <div>
              <h2>Nagpur Risk Map</h2>
              <p>
                {locations.length} monitored locations from PostgreSQL
              </p>
            </div>
          </div>

          <div className="map-placeholder">
            <div className="map-grid" />

            {risks.map((risk) => (
              <button
                key={risk.id}
                type="button"
                className={`map-marker ${risk.risk_level.toLowerCase()}`}
                style={markerPosition(risk)}
                title={`${risk.name} — ${risk.risk_score}`}
                onClick={() => setSelectedId(risk.id)}
              >
                <span>{risk.id}</span>
              </button>
            ))}

            <div className="map-caption">
              <strong>Nagpur monitoring network</strong>
              <span>
                Map layer placeholder — Leaflet will be added next.
              </span>
            </div>
          </div>
        </div>

        <aside className="details-panel">
          <div className="panel-heading">
            <div>
              <h2>Selected Location</h2>
              <p>Risk assessment</p>
            </div>
          </div>

          {selectedRisk && selectedLocation ? (
            <div className="details-content">
              <div className="selected-title">
                <span
                  className={`risk-pill ${selectedRisk.risk_level.toLowerCase()}`}
                >
                  {selectedRisk.risk_level}
                </span>
                <h3>{selectedRisk.name}</h3>
              </div>

              <div className="score-block">
                <span className="score-label">Risk score</span>
                <strong>{selectedRisk.risk_score.toFixed(2)}</strong>
              </div>

              <div className="detail-grid">
                <DetailItem
                  label="Police officers"
                  value={selectedRisk.police_officers}
                />
                <DetailItem
                  label="Latitude"
                  value={selectedLocation.latitude}
                />
                <DetailItem
                  label="Longitude"
                  value={selectedLocation.longitude}
                />
                <DetailItem
                  label="Coordinate source"
                  value={selectedLocation.coordinate_source}
                />
              </div>

              <button
                type="button"
                className="primary-button"
                onClick={() => setSelectedId(selectedRisk.id)}
              >
                View location
              </button>
            </div>
          ) : (
            <p className="empty-message">Select a location from the map.</p>
          )}
        </aside>
      </section>

      <section className="table-panel">
        <div className="panel-heading">
          <div>
            <h2>Location Risk Overview</h2>
            <p>Sorted by risk level and score</p>
          </div>
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Location</th>
                <th>Risk</th>
                <th>Score</th>
                <th>Police</th>
                <th>Coordinates</th>
              </tr>
            </thead>
            <tbody>
              {sortedRisks.map((risk) => {
                const location = locations.find((item) => item.id === risk.id)

                return (
                  <tr key={risk.id}>
                    <td>
                      <button
                        type="button"
                        className="location-button"
                        onClick={() => setSelectedId(risk.id)}
                      >
                        {risk.name}
                      </button>
                    </td>
                    <td>
                      <span
                        className={`risk-pill ${risk.risk_level.toLowerCase()}`}
                      >
                        {risk.risk_level}
                      </span>
                    </td>
                    <td>{risk.risk_score.toFixed(2)}</td>
                    <td>{risk.police_officers}</td>
                    <td>
                      {location
                        ? `${location.latitude.toFixed(4)}, ${location.longitude.toFixed(4)}`
                        : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="footer">
        <span>TrafficGuard AI</span>
        <span>
          Traffic measurements and risk inputs are synthetic demonstration
          data.
        </span>
      </footer>
    </main>
  )
}

function SummaryCard({ label, value, tone }) {
  return (
    <article className={`summary-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  )
}

function DetailItem({ label, value }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function markerPosition(risk) {
  const x = 12 + ((risk.longitude - 79.0325) / (79.136 - 79.0325)) * 76
  const y = 82 - ((risk.latitude - 21.0922) / (21.1857 - 21.0922)) * 68

  return {
    left: `${Math.max(6, Math.min(94, x))}%`,
    top: `${Math.max(8, Math.min(88, y))}%`,
  }
}

export default App