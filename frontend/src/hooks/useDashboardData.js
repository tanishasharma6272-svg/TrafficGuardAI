import { useCallback, useEffect, useState } from 'react'
import { fetchLocations, fetchMLRiskOverview } from '../services/api'
import { DATA_MODES } from '../constants/risk'

export function useDashboardData() {
  const [locations, setLocations] = useState([])
  const [risks, setRisks] = useState([])
  const [modelType, setModelType] = useState('BaselineRidge')
  const [trainingDataMode, setTrainingDataMode] = useState('SYNTHETIC_DEVELOPMENT')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)
  const [dataMode] = useState(DATA_MODES.DEMO)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const refetch = useCallback(() => {
    setLoading(true)
    setError('')
    setRefreshTrigger((k) => k + 1)
  }, [])

  useEffect(() => {
    let ignore = false

    Promise.all([fetchLocations(), fetchMLRiskOverview()])
      .then(([locationsData, mlRiskData]) => {
        if (!ignore) {
          setLocations(locationsData)
          setRisks(mlRiskData)
          if (mlRiskData.length > 0) {
            if (mlRiskData[0].model_type) {
              setModelType(mlRiskData[0].model_type)
            }
            if (mlRiskData[0].training_data_mode) {
              setTrainingDataMode(mlRiskData[0].training_data_mode)
            }
          }
          setLastUpdated(new Date())
          setError('')
          setLoading(false)
        }
      })
      .catch((requestError) => {
        if (!ignore) {
          const message =
            requestError.response?.data?.detail ||
            requestError.message ||
            'Unable to load ML risk overview from backend.'
          setError(message)
          setLoading(false)
        }
      })

    return () => {
      ignore = true
    }
  }, [refreshTrigger])

  return {
    locations,
    risks,
    modelType,
    trainingDataMode,
    loading,
    error,
    lastUpdated,
    dataMode,
    refetch,
  }
}
