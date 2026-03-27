import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = (() => {
  if (typeof window === 'undefined') return 'ws://localhost:8000/stream'

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${protocol}://${window.location.host}/stream`
})()
const MAX_ALERTS = 500

export function useAlertStream() {
  const [alerts, setAlerts]       = useState([])
  const [drifts, setDrifts]       = useState([])
  const [connected, setConnected] = useState(false)
  const [stats, setStats]         = useState(null)
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)
  const shouldReconnectRef = useRef(true)

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }
  }, [])

  const connect = useCallback(() => {
    const current = wsRef.current
    if (
      current?.readyState === WebSocket.OPEN ||
      current?.readyState === WebSocket.CONNECTING
    ) {
      return
    }

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (wsRef.current !== ws || !shouldReconnectRef.current) {
        ws.close()
        return
      }

      setConnected(true)
      clearReconnectTimer()
    }

    ws.onmessage = (e) => {
      if (wsRef.current !== ws) return

      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'ping') return

        if (msg.type === 'alert') {
          setAlerts(prev => {
            const next = [{ ...msg, id: Date.now() + Math.random() }, ...prev]
            return next.slice(0, MAX_ALERTS)
          })
        } else if (msg.type === 'drift') {
          setDrifts(prev => [{ ...msg, id: Date.now() }, ...prev].slice(0, 100))
        }
      } catch (_) {}
    }

    ws.onclose = () => {
      if (wsRef.current !== ws) return

      wsRef.current = null
      setConnected(false)
      clearReconnectTimer()

      if (!shouldReconnectRef.current) return

      reconnectTimer.current = setTimeout(() => {
        reconnectTimer.current = null
        connect()
      }, 3000)
    }

    ws.onerror = () => {
      if (wsRef.current === ws) ws.close()
    }
  }, [clearReconnectTimer])

  useEffect(() => {
    shouldReconnectRef.current = true
    connect()

    // Poll stats every 5s
    const statsInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/stats')
        const data = await res.json()
        setStats(data)
      } catch (_) {}
    }, 5000)

    return () => {
      shouldReconnectRef.current = false
      clearReconnectTimer()
      wsRef.current?.close()
      clearInterval(statsInterval)
    }
  }, [clearReconnectTimer, connect])

  // Derived metrics
  const attackAlerts  = alerts.filter(a => a.prediction === 1)
  const benignAlerts  = alerts.filter(a => a.prediction === 0)
  const attackRate    = alerts.length ? (attackAlerts.length / alerts.length * 100).toFixed(1) : '0.0'
  const avgConf       = alerts.length
    ? (alerts.slice(0, 50).reduce((s, a) => s + a.confidence, 0) / Math.min(50, alerts.length) * 100).toFixed(1)
    : '0.0'
  const acceptedDrifts = drifts.filter(d => d.accepted)
  const latestDrift = drifts[0] ?? null
  const acceptanceRate = drifts.length
    ? ((acceptedDrifts.length / drifts.length) * 100).toFixed(0)
    : '0'

  return {
    alerts, drifts, connected, stats,
    attackAlerts, benignAlerts,
    attackRate, avgConf,
    acceptedDrifts, latestDrift, acceptanceRate,
  }
}
