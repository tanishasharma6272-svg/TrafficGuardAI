import { useEffect, useMemo, useState } from 'react'
import { fetchMLLocationRisk, fetchRiskExplanation } from '../services/api'
import { buildLocationMap } from '../utils/riskStats'

export function useRiskSelection(locations, risks) {
  const [userSelectedId, setUserSelectedId] = useState(null)
  const [selectedDetail, setSelectedDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')

  const [selectedShap, setSelectedShap] = useState(null)
  const [shapLoading, setShapLoading] = useState(false)
  const [shapError, setShapError] = useState('')

  const locationMap = useMemo(() => buildLocationMap(locations), [locations])

  // Derive the active selectedId: user choice, or fallback to first available risk
  const selectedId = useMemo(() => {
    if (userSelectedId !== null) return userSelectedId
    return risks.length > 0 ? risks[0].id : null
  }, [userSelectedId, risks])

  // Fetch detailed ML risk breakdown and SHAP explainability independently when selection changes
  useEffect(() => {
    if (selectedId === null) {
      return
    }

    let active = true

    fetchMLLocationRisk(selectedId)
      .then((detail) => {
        if (active) {
          setSelectedDetail(detail)
          setDetailError('')
          setDetailLoading(false)
        }
      })
      .catch((err) => {
        if (active) {
          const msg =
            err.response?.data?.detail ||
            err.message ||
            'Failed to load ML location risk breakdown.'
          setDetailError(msg)
          setDetailLoading(false)
        }
      })

    fetchRiskExplanation(selectedId)
      .then((shapData) => {
        if (active) {
          setSelectedShap(shapData)
          setShapError('')
          setShapLoading(false)
        }
      })
      .catch((err) => {
        if (active) {
          const msg =
            err.response?.data?.detail ||
            err.message ||
            'Failed to compute SHAP feature attributions.'
          setShapError(msg)
          setShapLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [selectedId])

  const selectedRisk = useMemo(() => {
    return risks.find((r) => r.id === selectedId) || null
  }, [risks, selectedId])

  const selectedLocation = useMemo(() => {
    return selectedId !== null ? locationMap.get(selectedId) || null : null
  }, [locationMap, selectedId])

  return {
    selectedId,
    setSelectedId: setUserSelectedId,
    selectedRisk,
    selectedLocation,
    selectedDetail,
    detailLoading,
    detailError,
    selectedShap,
    shapLoading,
    shapError,
    locationMap,
  }
}
