import React from 'react'

function fmtDelta(value, digits = 3) {
  if (!Number.isFinite(value)) return '--'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

function Metric({ label, value, positive }) {
  return (
    <div
      style={{
        padding: '10px 12px',
        border: '1px solid var(--border)',
        borderRadius: 6,
        background: 'linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02))',
        minHeight: 68,
      }}
    >
      <div
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          letterSpacing: '0.12em',
          textTransform: 'uppercase',
          color: 'var(--text-dim)',
        }}
      >
        {label}
      </div>
      <div
        style={{
          marginTop: 8,
          fontFamily: 'var(--display)',
          fontSize: 22,
          color: positive ? 'var(--green)' : 'var(--amber)',
        }}
      >
        {value}
      </div>
    </div>
  )
}

export default function AdaptationSummary({ drifts, stats }) {
  const latest = drifts[0]
  const accepted = drifts.filter(d => d.accepted).length
  const attempted = drifts.length
  const acceptanceRate = attempted ? `${((accepted / attempted) * 100).toFixed(0)}%` : '--'

  if (!latest) {
    return (
      <div
        style={{
          padding: 16,
          borderBottom: '1px solid var(--border)',
          color: 'var(--text-dim)',
          fontSize: 11,
        }}
      >
        Waiting for the first adaptation attempt. Once drift is evaluated, this panel will show holdout quality, loss changes, and acceptance rate.
      </div>
    )
  }

  const holdoutDelta = (latest.after_f1 ?? 0) - (latest.before_f1 ?? 0)
  const supportLossDelta = (latest.support_loss_after ?? 0) - (latest.support_loss_before ?? 0)
  const holdoutLossDelta = (latest.val_loss_after ?? 0) - (latest.val_loss_before ?? 0)
  const tuneDelta = (latest.tune_f1_after ?? 0) - (latest.tune_f1_before ?? 0)

  return (
    <div
      style={{
        padding: 12,
        borderBottom: '1px solid var(--border)',
        display: 'grid',
        gap: 10,
        background: 'linear-gradient(180deg, rgba(0,255,163,0.05), rgba(0,0,0,0))',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 12,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: 'var(--text-dim)',
            }}
          >
            Latest adaptation quality
          </div>
          <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-bright)' }}>
            Window #{latest.window_id} via {latest.detector} {latest.accepted ? 'accepted' : 'rejected'}
          </div>
          {latest.trigger_reason && (
            <div style={{ marginTop: 4, fontSize: 10, color: 'var(--text-dim)' }}>
              {latest.trigger_reason}
            </div>
          )}
        </div>

        <div
          style={{
            padding: '5px 9px',
            borderRadius: 999,
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: latest.accepted ? 'var(--green)' : 'var(--amber)',
            background: latest.accepted ? 'var(--green-dim)' : 'var(--amber-dim)',
            border: `1px solid ${latest.accepted ? 'rgba(0,255,163,0.3)' : 'rgba(255,169,77,0.3)'}`,
            whiteSpace: 'nowrap',
          }}
        >
          v{latest.model_version} · {latest.adapt_steps} steps
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: 8,
        }}
      >
        <Metric label="Acceptance rate" value={acceptanceRate} positive={accepted > 0} />
        <Metric label="Holdout F1 delta" value={fmtDelta(holdoutDelta)} positive={holdoutDelta >= 0} />
        <Metric label="Support loss delta" value={fmtDelta(supportLossDelta, 4)} positive={supportLossDelta <= 0} />
        <Metric label="Holdout loss delta" value={fmtDelta(holdoutLossDelta, 4)} positive={holdoutLossDelta <= 0} />
      </div>

      <div
        style={{
          display: 'flex',
          gap: 10,
          flexWrap: 'wrap',
          fontFamily: 'var(--mono)',
          fontSize: 10,
          color: 'var(--text-dim)',
        }}
      >
        <span style={{ padding: '3px 8px', background: 'var(--bg2)', borderRadius: 999 }}>
          Support F1 {latest.support_f1_before?.toFixed(3)} {'->'} {latest.support_f1_after?.toFixed(3)}
        </span>
        <span style={{ padding: '3px 8px', background: 'var(--bg2)', borderRadius: 999 }}>
          Holdout F1 {latest.before_f1?.toFixed(3)} {'->'} {latest.after_f1?.toFixed(3)}
        </span>
        <span style={{ padding: '3px 8px', background: 'var(--bg2)', borderRadius: 999 }}>
          Tuning F1 delta {fmtDelta(tuneDelta)}
        </span>
        <span style={{ padding: '3px 8px', background: 'var(--bg2)', borderRadius: 999 }}>
          Trigger score {latest.trigger_score?.toFixed(3) ?? '--'}
        </span>
        <span style={{ padding: '3px 8px', background: 'var(--bg2)', borderRadius: 999 }}>
          {stats?.hypotheses_accepted ?? accepted} accepted / {stats?.hypotheses_tried ?? attempted} tried
        </span>
      </div>
    </div>
  )
}
