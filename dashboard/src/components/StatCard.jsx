import React from 'react'

const styles = {
  card: {
    background: 'var(--bg2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '16px 20px',
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    position: 'relative',
    overflow: 'hidden',
  },
  accent: {
    position: 'absolute',
    top: 0, left: 0, right: 0,
    height: 2,
  },
  label: {
    fontFamily: 'var(--mono)',
    fontSize: 10,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: 'var(--text-dim)',
  },
  value: {
    fontFamily: 'var(--display)',
    fontSize: 28,
    fontWeight: 700,
    lineHeight: 1,
    color: 'var(--text-bright)',
  },
  sub: {
    fontSize: 11,
    color: 'var(--text-dim)',
    marginTop: 2,
  },
}

const COLORS = {
  green: 'var(--green)',
  red:   'var(--red)',
  amber: 'var(--amber)',
  blue:  'var(--blue)',
}

export default function StatCard({ label, value, sub, color = 'green', blink = false }) {
  const accentColor = COLORS[color] || color
  return (
    <div style={styles.card}>
      <div style={{ ...styles.accent, background: accentColor, opacity: 0.7 }} />
      <div style={styles.label}>{label}</div>
      <div style={{
        ...styles.value,
        color: accentColor,
        animation: blink ? 'pulse-green 2s ease-in-out infinite' : 'none',
      }}>
        {value}
      </div>
      {sub && <div style={styles.sub}>{sub}</div>}
    </div>
  )
}