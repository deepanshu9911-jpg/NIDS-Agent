import React, { useRef, useEffect } from 'react'

const ROW = {
  display: 'grid',
  gridTemplateColumns: '80px 60px 1fr 80px 90px',
  gap: 12,
  padding: '7px 14px',
  alignItems: 'center',
  borderBottom: '1px solid var(--border)',
  fontFamily: 'var(--mono)',
  fontSize: 11,
  animation: 'slide-in 0.2s ease',
}

function ts(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString('en-GB', { hour12: false })
}

function ConfBar({ value }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{
        height: 4, width: 60, background: 'var(--bg3)', borderRadius: 2, overflow: 'hidden'
      }}>
        <div style={{
          height: '100%',
          width: `${(value * 100).toFixed(0)}%`,
          background: value > 0.8 ? 'var(--green)' : value > 0.5 ? 'var(--amber)' : 'var(--red)',
          borderRadius: 2,
          transition: 'width 0.3s',
        }} />
      </div>
      <span style={{ color: 'var(--text-dim)', fontSize: 10 }}>{(value * 100).toFixed(0)}%</span>
    </div>
  )
}

export default function AlertFeed({ alerts }) {
  const listRef = useRef(null)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      {/* Header */}
      <div style={{
        ...ROW,
        animation: 'none',
        borderBottom: '1px solid var(--border2)',
        color: 'var(--text-dim)',
        fontSize: 9,
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        background: 'var(--bg1)',
        padding: '8px 14px',
      }}>
        <span>Time</span>
        <span>Window</span>
        <span>Status</span>
        <span>Conf.</span>
        <span>Model v</span>
      </div>

      {/* Rows */}
      <div ref={listRef} style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
        {alerts.length === 0 && (
          <div style={{ padding: '24px 14px', color: 'var(--text-dim)', textAlign: 'center' }}>
            Waiting for stream…
            <span style={{ animation: 'blink 1s step-end infinite', marginLeft: 2 }}>_</span>
          </div>
        )}
        {alerts.map(a => (
          <div key={a.id} style={{
            ...ROW,
            background: a.prediction === 1 ? 'rgba(255,77,106,0.04)' : 'transparent',
          }}>
            <span style={{ color: 'var(--text-dim)' }}>{ts(a.timestamp)}</span>
            <span style={{ color: 'var(--blue)' }}>#{a.window_id}</span>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              color: a.prediction === 1 ? 'var(--red)' : 'var(--green)',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: a.prediction === 1 ? 'var(--red)' : 'var(--green)',
                display: 'inline-block',
                boxShadow: a.prediction === 1
                  ? '0 0 6px var(--red)'
                  : '0 0 6px var(--green)',
              }} />
              {a.prediction === 1 ? 'ATTACK' : 'BENIGN'}
            </span>
            <ConfBar value={a.confidence} />
            <span style={{ color: 'var(--text-dim)' }}>v{a.model_version}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
