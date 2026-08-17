import { UnavailableNotice } from '../shared/UnavailableNotice'

export function RiskTrends() {
  return (
    <section className="risk-trends-panel" aria-label="Risk Trends and Telemetry Analytics">
      <div className="panel-header">
        <div>
          <h2 className="panel-title">Risk Trends &amp; Telemetry Analytics</h2>
          <p className="panel-subtitle">Temporal variance and 24-hour diurnal risk curve</p>
        </div>
      </div>

      <div className="trends-body">
        <UnavailableNotice
          title="Time-Series Analytics — Awaiting historical / live telemetry feed"
          description="Diurnal risk curves, hourly congestion histograms, and zone-based temporal trends require time-series sensor ingest. Current PostgreSQL store holds steady-state demo records."
          badge="HISTORICAL FEED PENDING"
          targetEndpoint="GET /api/telemetry/trends"
        >
          <div className="trends-placeholder-graphic" aria-hidden="true">
            <div className="trend-bar-chart-mock">
              <div className="mock-bar" style={{ height: '35%' }} />
              <div className="mock-bar" style={{ height: '48%' }} />
              <div className="mock-bar" style={{ height: '70%' }} />
              <div className="mock-bar" style={{ height: '90%' }} />
              <div className="mock-bar" style={{ height: '85%' }} />
              <div className="mock-bar" style={{ height: '60%' }} />
              <div className="mock-bar" style={{ height: '40%' }} />
              <div className="mock-bar" style={{ height: '25%' }} />
            </div>
            <div className="mock-axis-labels">
              <span>06:00</span>
              <span>09:00 (Peak)</span>
              <span>12:00</span>
              <span>18:00 (Peak)</span>
              <span>22:00</span>
            </div>
          </div>
        </UnavailableNotice>
      </div>
    </section>
  )
}
