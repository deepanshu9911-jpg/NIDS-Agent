import React from 'react'

function ts(t) {
  return new Date(t * 1000).toLocaleTimeString('en-GB', { hour12: false })
}

function delta(before, after, digits = 3) {
  const value = (after ?? 0) - (before ?? 0)
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(digits)}`
}

export default function DriftLog({ drifts }) {
  if (!drifts.length) {
    return (
      <div style={{ padding: 16, color: 'var(--text-dim)', fontSize: 11 }}>
        No drift events yet.
      </div>
    )
  }

  return (
    <div style={{ overflowY: 'auto', maxHeight: '100%', minHeight: 0 }}>
      {drifts.map(d => (
        <div
          key={d.id}
          style={{
            padding: '12px 14px',
            borderBottom: '1px solid var(--border)',
            display: 'grid',
            gap: 10,
            animation: 'fade-up 0.3s ease',
            background: d.accepted ? 'rgba(0,255,163,0.035)' : 'rgba(255,169,77,0.035)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <div style={{ minWidth: 0 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-dim)' }}>
                {ts(d.timestamp)}
              </span>
              <div style={{ marginTop: 6, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <span
                  style={{
                    fontSize: 10,
                    padding: '2px 8px',
                    borderRadius: 999,
                    fontFamily: 'var(--mono)',
                    background: d.accepted ? 'var(--green-dim)' : 'var(--amber-dim)',
                    color: d.accepted ? 'var(--green)' : 'var(--amber)',
                    border: `1px solid ${d.accepted ? 'rgba(0,255,163,0.3)' : 'rgba(255,169,77,0.3)'}`,
                  }}
                >
                  {d.accepted ? 'ADAPTED' : 'REJECTED'}
                </span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-dim)' }}>
                  v{d.model_version} · {d.adapt_steps} steps
                </span>
              </div>
              <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-dim)' }}>
                Window #{d.window_id} · {d.detector}
              </div>
              {d.trigger_reason && (
                <div style={{ marginTop: 4, fontSize: 10, color: 'var(--text-dim)' }}>
                  {d.trigger_reason}
                </div>
              )}
            </div>

            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-dim)', textAlign: 'right' }}>
              <div>Action: {d.action}</div>
              <div style={{ marginTop: 4 }}>Holdout delta {delta(d.before_f1, d.after_f1)}</div>
              <div style={{ marginTop: 4 }}>Trigger {Number.isFinite(d.trigger_score) ? d.trigger_score.toFixed(3) : '--'}</div>
            </div>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: 10,
              fontSize: 10,
              fontFamily: 'var(--mono)',
            }}
          >
            <div style={{ padding: '8px 10px', background: 'var(--bg2)', borderRadius: 6 }}>
              <div style={{ color: 'var(--text-dim)' }}>Support</div>
              <div style={{ marginTop: 4 }}>
                <span style={{ color: 'var(--red)' }}>{(d.support_f1_before ?? 0).toFixed(3)}</span>
                {' -> '}
                <span style={{ color: 'var(--green)' }}>{(d.support_f1_after ?? 0).toFixed(3)}</span>
              </div>
              <div style={{ marginTop: 4, color: 'var(--text-dim)' }}>
                Loss {delta(d.support_loss_before, d.support_loss_after, 4)}
              </div>
            </div>

            <div style={{ padding: '8px 10px', background: 'var(--bg2)', borderRadius: 6 }}>
              <div style={{ color: 'var(--text-dim)' }}>Holdout</div>
              <div style={{ marginTop: 4 }}>
                <span style={{ color: 'var(--red)' }}>{(d.before_f1 ?? 0).toFixed(3)}</span>
                {' -> '}
                <span style={{ color: 'var(--green)' }}>{(d.after_f1 ?? 0).toFixed(3)}</span>
              </div>
              <div style={{ marginTop: 4, color: 'var(--text-dim)' }}>
                Loss {delta(d.val_loss_before, d.val_loss_after, 4)}
              </div>
            </div>

            <div style={{ padding: '8px 10px', background: 'var(--bg2)', borderRadius: 6 }}>
              <div style={{ color: 'var(--text-dim)' }}>Tuning</div>
              <div style={{ marginTop: 4 }}>
                <span style={{ color: 'var(--red)' }}>{(d.tune_f1_before ?? 0).toFixed(3)}</span>
                {' -> '}
                <span style={{ color: 'var(--green)' }}>{(d.tune_f1_after ?? 0).toFixed(3)}</span>
              </div>
              <div style={{ marginTop: 4, color: 'var(--text-dim)' }}>
                Delta {delta(d.tune_f1_before, d.tune_f1_after)}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
