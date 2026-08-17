import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { LEVEL_COLOR, NAGPUR_BOUNDS } from '../../constants/risk'
import { RiskLegend } from './RiskLegend'

export function RiskMap({
  locations,
  sortedRisks,
  selectedId,
  onSelectLocation,
  selectedLocation,
  deploymentResult,
}) {
  const mapContainerRef = useRef(null)
  const mapInstanceRef = useRef(null)
  const markersLayerRef = useRef(null)
  const deploymentLayerRef = useRef(null)

  // Initialize Leaflet map instance once
  useEffect(() => {
    if (mapInstanceRef.current || !mapContainerRef.current) return

    const map = L.map(mapContainerRef.current, {
      zoomControl: false,
      attributionControl: true,
    }).fitBounds(NAGPUR_BOUNDS, { padding: [24, 24] })

    L.control.zoom({ position: 'bottomright' }).addTo(map)

    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
      {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
      },
    ).addTo(map)

    deploymentLayerRef.current = L.layerGroup().addTo(map)
    markersLayerRef.current = L.layerGroup().addTo(map)
    mapInstanceRef.current = map

    // Handle container dimension changes to prevent 0x0 canvas glitches
    const invalidate = () => map.invalidateSize()
    requestAnimationFrame(invalidate)
    const settleTimer = setTimeout(invalidate, 300)
    window.addEventListener('resize', invalidate)

    let resizeObserver
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(invalidate)
      resizeObserver.observe(mapContainerRef.current)
    }

    return () => {
      clearTimeout(settleTimer)
      window.removeEventListener('resize', invalidate)
      resizeObserver?.disconnect()
      map.remove()
      mapInstanceRef.current = null
    }
  }, [])

  // Redraw monitored risk markers whenever the filtered/sorted set or selection changes
  useEffect(() => {
    const map = mapInstanceRef.current
    const layer = markersLayerRef.current
    if (!map || !layer) return

    layer.clearLayers()

    sortedRisks.forEach(({ risk, location }) => {
      if (!location) return

      const isSelected = risk.id === selectedId
      const pulse = risk.risk_level === 'Critical' || risk.risk_level === 'High'
      const color = LEVEL_COLOR[risk.risk_level] || '#8c84a8'

      const icon = L.divIcon({
        className: 'marker-wrapper',
        html: `
          <div class="marker-pulse ${pulse ? 'is-pulse-active' : ''} ${isSelected ? 'is-selected' : ''}" style="--marker-color:${color}">
            ${pulse ? '<span class="pulse-ring"></span><span class="pulse-ring pulse-ring--delay"></span>' : ''}
            <span class="marker-core"></span>
          </div>
        `,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      })

      const marker = L.marker([location.latitude, location.longitude], { icon })
        .bindTooltip(
          `<strong>${risk.name}</strong><br/>ML Risk Score: ${risk.risk_score.toFixed(2)} (${risk.risk_level})<br/>Stationed Police: ${risk.police_officers} units`,
          {
            direction: 'top',
            offset: [0, -12],
            className: 'tg-map-tooltip',
          },
        )
        .on('click', () => onSelectLocation(risk.id))

      marker.addTo(layer)
    })
  }, [sortedRisks, selectedId, onSelectLocation])

  // Draw distinct deployment units and patrol coverage radii when optimizer results exist
  useEffect(() => {
    const map = mapInstanceRef.current
    const deployLayer = deploymentLayerRef.current
    if (!map || !deployLayer) return

    deployLayer.clearLayers()

    if (!deploymentResult?.selected_units || deploymentResult.selected_units.length === 0) {
      return
    }

    const radiusMeters = (deploymentResult.coverage_radius_km || 2.0) * 1000

    deploymentResult.selected_units.forEach((unit) => {
      const isSelected = unit.location_id === selectedId

      // Draw patrol radius circle
      const circle = L.circle([unit.latitude, unit.longitude], {
        radius: radiusMeters,
        color: '#4f46e5',
        weight: 1.5,
        dashArray: '4, 6',
        fillColor: '#6366f1',
        fillOpacity: 0.08,
      })
      circle.addTo(deployLayer)

      // Draw visually distinct deployment hub marker
      const deployIcon = L.divIcon({
        className: 'deploy-marker-wrapper',
        html: `
          <div class="deploy-hub-marker ${isSelected ? 'is-selected' : ''}">
            <div class="deploy-hub-badge mono">#${unit.rank}</div>
            <div class="deploy-hub-shield">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3z"/>
              </svg>
            </div>
          </div>
        `,
        iconSize: [36, 36],
        iconAnchor: [18, 18],
      })

      const deployMarker = L.marker([unit.latitude, unit.longitude], {
        icon: deployIcon,
        zIndexOffset: 500,
      })
        .bindTooltip(
          `
          <div class="deploy-map-tooltip">
            <span class="deploy-tooltip-badge">POLICE DEPLOYMENT · RANK #${unit.rank}</span>
            <strong class="deploy-tooltip-name">${unit.location_name}</strong>
            <div class="deploy-tooltip-row">ML Risk Score: <strong>${unit.risk_score.toFixed(2)} (${unit.risk_level})</strong></div>
            <div class="deploy-tooltip-row">Covered Nodes: <strong>${unit.covered_location_count} intersections</strong></div>
            <div class="deploy-tooltip-row">Covered Risk Mass: <strong>${unit.covered_risk_score.toFixed(1)} pts</strong></div>
          </div>
          `,
          {
            direction: 'top',
            offset: [0, -16],
            className: 'tg-map-tooltip tg-map-tooltip--deployment',
          },
        )
        .on('click', () => onSelectLocation(unit.location_id))

      deployMarker.addTo(deployLayer)
    })
  }, [deploymentResult, selectedId, onSelectLocation])

  // Fly to the selected location on change
  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || !selectedLocation) return

    map.flyTo([selectedLocation.latitude, selectedLocation.longitude], 14, {
      duration: 0.6,
    })
  }, [selectedLocation])

  return (
    <div className="map-panel">
      <div className="panel-header">
        <div>
          <h2 className="panel-title">Nagpur Risk Map</h2>
          <p className="panel-subtitle">
            {locations.length} monitored intersections from PostgreSQL demo store
            {deploymentResult?.selected_units?.length > 0 && (
              <span className="map-deployment-active-tag">
                · {deploymentResult.selected_units.length} Deployment Hubs Active
              </span>
            )}
          </p>
        </div>
      </div>

      <div className="map-viewport-frame">
        <div
          ref={mapContainerRef}
          className="map-canvas-element"
          style={{ width: '100%', height: '100%' }}
        />

        <RiskLegend
          totalMonitored={locations.length}
          filteredCount={sortedRisks.length}
          deploymentActive={Boolean(deploymentResult?.selected_units?.length)}
        />
      </div>
    </div>
  )
}
