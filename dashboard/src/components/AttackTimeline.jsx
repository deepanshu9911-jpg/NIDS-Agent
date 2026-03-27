import React, { useMemo } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

function buildBuckets(alerts, drifts, bucketCount = 60) {
  if (!alerts.length) return []

  const now = Date.now() / 1000
  const oldest = now - 120
  const bucket = (now - oldest) / bucketCount

  const bins = Array.from({ length: bucketCount }, (_, i) => ({
    t: Math.round(oldest + i * bucket),
    attacks: 0,
    benign: 0,
    drift: false,
  }))

  for (const alert of alerts) {
    const idx = Math.floor((alert.timestamp - oldest) / bucket)
    if (idx < 0 || idx >= bucketCount) continue
    if (alert.prediction === 1) bins[idx].attacks += 1
    else bins[idx].benign += 1
  }

  const driftTs = new Set(drifts.map(drift => Math.floor((drift.timestamp - oldest) / bucket)))
  for (const idx of driftTs) {
    if (idx >= 0 && idx < bucketCount) bins[idx].drift = true
  }

  return bins
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload[0]?.payload

  return (
    <div
      style={{
        background: 'var(--bg2)',
        border: '1px solid var(--border2)',
        padding: '8px 12px',
        borderRadius: 4,
        fontSize: 11,
        fontFamily: 'var(--mono)',
        boxShadow: '0 12px 30px rgba(0,0,0,0.2)',
      }}
    >
      <div style={{ color: 'var(--red)' }}>Attacks: {point?.attacks}</div>
      <div style={{ color: 'var(--green)' }}>Benign: {point?.benign}</div>
      {point?.drift && <div style={{ color: 'var(--amber)' }}>Drift detected</div>}
    </div>
  )
}

export default function AttackTimeline({ alerts, drifts }) {
  const data = useMemo(() => buildBuckets(alerts, drifts), [alerts, drifts])
  const driftLines = data.map((point, i) => ({ i, drift: point.drift })).filter(point => point.drift)

  return (
    <div style={{ width: '100%', height: 160 }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="gAtk" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ff4d6a" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#ff4d6a" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gBgn" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00ffa3" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#00ffa3" stopOpacity={0} />
            </linearGradient>
          </defs>

          <XAxis dataKey="t" hide />
          <YAxis tick={{ fontSize: 9, fill: 'var(--text-dim)', fontFamily: 'var(--mono)' }} />
          <Tooltip content={<CustomTooltip />} />

          {driftLines.map(({ i }) => (
            <ReferenceLine
              key={i}
              x={data[i]?.t}
              stroke="var(--amber)"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
          ))}

          <Area
            type="monotone"
            dataKey="benign"
            stroke="var(--green)"
            strokeWidth={1.5}
            fill="url(#gBgn)"
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="attacks"
            stroke="var(--red)"
            strokeWidth={1.5}
            fill="url(#gAtk)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
