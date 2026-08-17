import { UnavailableNotice } from '../shared/UnavailableNotice'

export function ShapPlaceholder() {
  return (
    <div className="shap-explanation-container">
      <UnavailableNotice
        title="AI Explanation (SHAP) — Unavailable in current demo API"
        description="Shapley Additive Explanations (SHAP) feature attribution values will be computed in real-time when the ML neural risk model inference pipeline is deployed."
        badge="ML EXPLAINABILITY PENDING"
        targetEndpoint="GET /api/risk/{location_id}/shap"
      >
        <div className="shap-wireframe-preview" aria-hidden="true">
          <div className="wireframe-row">
            <span className="wireframe-label">Peak Congestion Velocity</span>
            <div className="wireframe-bar wireframe-bar--pos" style={{ width: '45%' }} />
          </div>
          <div className="wireframe-row">
            <span className="wireframe-label">Intersection Geometry Risk</span>
            <div className="wireframe-bar wireframe-bar--pos" style={{ width: '30%' }} />
          </div>
          <div className="wireframe-row">
            <span className="wireframe-label">Police Deterrence Factor</span>
            <div className="wireframe-bar wireframe-bar--neg" style={{ width: '20%' }} />
          </div>
        </div>
      </UnavailableNotice>
    </div>
  )
}
