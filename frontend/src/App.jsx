import { useCallback, useState } from 'react'
import './App.css'
import { useLiveClock } from './hooks/useLiveClock'
import { useDashboardData } from './hooks/useDashboardData'
import { useRiskSelection } from './hooks/useRiskSelection'
import { useRiskFilters } from './hooks/useRiskFilters'
import { useRiskSort } from './hooks/useRiskSort'
import { recommendDeployment } from './services/api'

import { LoadingScreen } from './components/shared/LoadingScreen'
import { ErrorScreen } from './components/shared/ErrorScreen'
import { Header } from './components/Header/Header'
import { ExecutiveSummary } from './components/ExecutiveSummary/ExecutiveSummary'
import { RiskMap } from './components/RiskMap/RiskMap'
import { LocationDetails } from './components/LocationDetails/LocationDetails'
import { HighRiskLocations } from './components/HighRiskLocations/HighRiskLocations'
import { DeploymentPanel } from './components/DeploymentPanel/DeploymentPanel'
import { CoverageComparison } from './components/CoverageComparison/CoverageComparison'
import { RiskTrends } from './components/RiskTrends/RiskTrends'
import { RiskTable } from './components/RiskTable/RiskTable'

function App() {
  const clock = useLiveClock()
  const {
    locations,
    risks,
    modelType,
    trainingDataMode,
    loading,
    error,
    lastUpdated,
    dataMode,
    refetch,
  } = useDashboardData()

  const {
    selectedId,
    setSelectedId,
    selectedRisk,
    selectedLocation,
    selectedDetail,
    detailLoading,
    detailError,
    selectedShap,
    shapLoading,
    shapError,
    locationMap,
  } = useRiskSelection(locations, risks)

  const {
    query,
    setQuery,
    activeLevel,
    setActiveLevel,
    clearFilters,
    hasActiveFilters,
    filteredRisks,
  } = useRiskFilters(risks)

  const { sortKey, sortDir, toggleSort, sortedRisks } = useRiskSort(
    filteredRisks,
    locationMap,
  )

  // Police Deployment Optimizer State
  const [deploymentResult, setDeploymentResult] = useState(null)
  const [deploymentLoading, setDeploymentLoading] = useState(false)
  const [deploymentError, setDeploymentError] = useState('')

  const handleExecuteDeployment = useCallback(async (params) => {
    setDeploymentLoading(true)
    setDeploymentError('')
    try {
      const result = await recommendDeployment(params)
      setDeploymentResult(result)
      setDeploymentLoading(false)
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Failed to compute police deployment recommendations.'
      setDeploymentError(msg)
      setDeploymentLoading(false)
    }
  }, [])

  if (loading) {
    return <LoadingScreen />
  }

  if (error) {
    return <ErrorScreen error={error} onRetry={refetch} />
  }

  return (
    <main className="app-shell">
      {/* 1. Command Topbar */}
      <Header
        clock={clock}
        lastUpdated={lastUpdated}
        dataMode={dataMode}
        modelType={modelType}
        trainingDataMode={trainingDataMode}
        onRefresh={refetch}
        loading={loading}
      />

      {/* 2. Executive Risk Summary */}
      <ExecutiveSummary
        risks={risks}
        activeLevel={activeLevel}
        onSelectLevel={setActiveLevel}
        onClearFilter={clearFilters}
        hasActiveFilter={hasActiveFilters}
      />

      {/* 3. Primary Command Workspace (Nagpur Map + ML Location Intelligence) */}
      <section className="command-workspace-grid">
        <RiskMap
          locations={locations}
          sortedRisks={sortedRisks}
          selectedId={selectedId}
          onSelectLocation={setSelectedId}
          selectedLocation={selectedLocation}
          deploymentResult={deploymentResult}
        />

        <LocationDetails
          selectedRisk={selectedRisk}
          selectedLocation={selectedLocation}
          selectedDetail={selectedDetail}
          detailLoading={detailLoading}
          detailError={detailError}
          selectedShap={selectedShap}
          shapLoading={shapLoading}
          shapError={shapError}
        />
      </section>

      {/* 4. Tactical Operations Grid (High-Risk Queue, Deployment Optimizer, Coverage, Trends) */}
      <section className="tactical-operations-grid">
        <HighRiskLocations
          risks={risks}
          selectedId={selectedId}
          onSelectLocation={setSelectedId}
        />

        <DeploymentPanel
          deploymentResult={deploymentResult}
          loading={deploymentLoading}
          error={deploymentError}
          onExecute={handleExecuteDeployment}
          selectedId={selectedId}
          onSelectLocation={setSelectedId}
        />

        <CoverageComparison
          deploymentResult={deploymentResult}
        />

        <RiskTrends />
      </section>

      {/* 5. Comprehensive Location Registry Table */}
      <RiskTable
        sortedRisks={sortedRisks}
        totalCount={risks.length}
        selectedId={selectedId}
        onSelectLocation={setSelectedId}
        query={query}
        setQuery={setQuery}
        activeLevel={activeLevel}
        setActiveLevel={setActiveLevel}
        sortKey={sortKey}
        sortDir={sortDir}
        onSort={toggleSort}
        onClearFilters={clearFilters}
        hasActiveFilters={hasActiveFilters}
      />

      {/* 6. Command Footer */}
      <footer className="command-footer">
        <div className="footer-left">
          <span>TrafficGuard AI · Nagpur Command</span>
          <span className="footer-divider">|</span>
          <span>PostgreSQL Demo Dataset (20 Intersections) · ML Inference ({modelType})</span>
        </div>
        <div className="footer-right">
          <span>Simulation &amp; Demo Mode Active — ML models trained on synthetic development data</span>
        </div>
      </footer>
    </main>
  )
}

export default App
