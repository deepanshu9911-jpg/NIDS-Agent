import React, { useEffect, useState } from 'react'
import { useAlertStream } from './hooks/useAlertStream.js'
import StatCard from './components/StatCard'
import AlertFeed from './components/AlertFeed'
import AttackTimeline from './components/AttackTimeline'
import EvolutionGraph from './components/EvolutionGraph'
import DriftLog from './components/DriftLog'
import AdaptationSummary from './components/AdaptationSummary'

const S = {
  shell: {
    display: 'grid',
    gridTemplateRows: 'auto auto 1fr',
    height: '100vh',
    overflow: 'hidden',
    background:
      'radial-gradient(circle at top right, rgba(77,166,255,0.08), transparent 28%), radial-gradient(circle at top left, rgba(0,255,163,0.08), transparent 24%), var(--bg)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
    padding: '12px 24px',
    borderBottom: '1px solid var(--border)',
    background: 'linear-gradient(180deg, rgba(13,17,23,0.96), rgba(13,17,23,0.88))',
  },
  logo: {
    fontFamily: 'var(--display)',
    fontWeight: 800,
    fontSize: 20,
    letterSpacing: '-0.02em',
    color: 'var(--green)',
    textShadow: '0 0 16px rgba(0,255,163,0.18)',
  },
  logoSub: {
    fontFamily: 'var(--mono)',
    fontSize: 9,
    color: 'var(--text-dim)',
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    marginTop: 2,
  },
  status: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
    fontFamily: 'var(--mono)',
    fontSize: 11,
    color: 'var(--text-dim)',
  },
  dot: connected => ({
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: connected ? 'var(--green)' : 'var(--red)',
    boxShadow: connected ? '0 0 8px var(--green)' : '0 0 8px var(--red)',
    animation: connected ? 'pulse-green 2s ease-in-out infinite' : 'none',
  }),
  statsRow: {
    display: 'grid',
    gap: 8,
    padding: '12px 16px',
    background: 'var(--bg1)',
    borderBottom: '1px solid var(--border)',
  },
  main: {
    display: 'grid',
    gap: 8,
    padding: 10,
    overflow: 'hidden',
  },
  panel: {
    background: 'var(--bg1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    boxShadow: '0 14px 40px rgba(0,0,0,0.22)',
  },
  panelHead: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    padding: '10px 14px',
    borderBottom: '1px solid var(--border)',
    fontFamily: 'var(--mono)',
    fontSize: 9,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: 'var(--text-dim)',
    flexShrink: 0,
  },
  dot2: color => ({
    width: 5,
    height: 5,
    borderRadius: '50%',
    background: color,
    display: 'inline-block',
    marginRight: 6,
  }),
}

function Panel({ title, badge, badgeColor = 'var(--green)', children, style = {} }) {
  return (
    <div style={{ ...S.panel, ...style }}>
      <div style={S.panelHead}>
        <span>
          <span style={S.dot2(badgeColor)} />
          {title}
        </span>
        {badge && (
          <span
            style={{
              background: 'var(--bg3)',
              padding: '2px 7px',
              borderRadius: 3,
              fontSize: 9,
              color: 'var(--text-dim)',
            }}
          >
            {badge}
          </span>
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {children}
      </div>
    </div>
  )
}

export default function App() {
  const {
    alerts,
    drifts,
    connected,
    stats,
    attackAlerts,
    attackRate,
    avgConf,
    acceptedDrifts,
    latestDrift,
    acceptanceRate,
  } = useAlertStream()

  const [activeTab, setActiveTab] = useState('alerts')
  const [viewportWidth, setViewportWidth] = useState(
    typeof window === 'undefined' ? 1440 : window.innerWidth
  )

  useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const isCompact = viewportWidth < 1200
  const isNarrow = viewportWidth < 900
  const statsColumns = isNarrow
    ? 'repeat(2, minmax(0, 1fr))'
    : isCompact
      ? 'repeat(3, minmax(0, 1fr))'
      : 'repeat(6, minmax(0, 1fr))'

  const mainLayout = isCompact
    ? {
        gridTemplateColumns: '1fr',
        gridTemplateRows: '340px minmax(420px, 1fr) minmax(460px, 1fr)',
        overflowY: 'auto',
      }
    : {
        gridTemplateColumns: '1fr 360px',
        gridTemplateRows: '1fr 1fr',
      }

  return (
    <div style={S.shell}>
      <header
        style={{
          ...S.header,
          flexDirection: isNarrow ? 'column' : 'row',
          alignItems: isNarrow ? 'flex-start' : 'center',
        }}
      >
        <div>
          <div style={S.logo}>NIDS - NEURAL THREAT INTEL</div>
          <div style={S.logoSub}>Self-evolving intrusion detection · GNN + MAML</div>
        </div>

        <div style={S.status}>
          <div style={S.dot(connected)} />
          {connected ? 'STREAM LIVE' : 'RECONNECTING...'}
          {stats && (
            <span style={{ marginLeft: 16, color: 'var(--text-dim)' }}>
              Model v{stats.model_version} · {stats.windows_processed.toLocaleString()} windows processed
            </span>
          )}
        </div>
      </header>

      <div style={{ ...S.statsRow, gridTemplateColumns: statsColumns }}>
        <StatCard label="Total windows" value={alerts.length.toLocaleString()} sub="processed" color="blue" />
        <StatCard
          label="Attack alerts"
          value={attackAlerts.length.toLocaleString()}
          sub={`${attackRate}% of traffic`}
          color="red"
          blink={attackAlerts.length > 0}
        />
        <StatCard
          label="Drift events"
          value={drifts.length}
          sub={`${acceptedDrifts.length} accepted`}
          color="amber"
        />
        <StatCard label="Avg confidence" value={`${avgConf}%`} sub="last 50 windows" color="green" />
        <StatCard
          label="Model version"
          value={stats?.model_version ?? 0}
          sub={`${stats?.hypotheses_accepted ?? 0} adaptations`}
          color="green"
        />
        <StatCard
          label="Acceptance rate"
          value={`${acceptanceRate}%`}
          sub={latestDrift ? `latest ${latestDrift.accepted ? 'accepted' : 'rejected'}` : 'awaiting drift'}
          color={acceptedDrifts.length ? 'green' : 'amber'}
        />
      </div>

      <div style={{ ...S.main, ...mainLayout }}>
        <Panel title="Traffic timeline" badge="live · 2min window" badgeColor="var(--green)">
          <div style={{ padding: '8px 8px 4px', flex: 1 }}>
            <AttackTimeline alerts={alerts} drifts={drifts} />
            <div
              style={{
                display: 'flex',
                gap: 16,
                paddingLeft: 8,
                marginTop: 6,
                flexWrap: 'wrap',
                fontSize: 9,
                fontFamily: 'var(--mono)',
                color: 'var(--text-dim)',
              }}
            >
              <span><span style={{ color: 'var(--green)' }}>■</span> Benign</span>
              <span><span style={{ color: 'var(--red)' }}>■</span> Attack</span>
              <span><span style={{ color: 'var(--amber)' }}>|</span> Drift event</span>
            </div>
          </div>
        </Panel>

        <Panel
          title="Model evolution graph"
          badge="D3 force"
          badgeColor="var(--amber)"
          style={isCompact ? {} : { gridRow: '1 / 3' }}
        >
          <div style={{ flex: 1, padding: 8 }}>
            <EvolutionGraph alerts={alerts} drifts={drifts} />
          </div>
          <div
            style={{
              padding: '8px 14px',
              borderTop: '1px solid var(--border)',
              fontSize: 9,
              color: 'var(--text-dim)',
              fontFamily: 'var(--mono)',
              display: 'flex',
              gap: 16,
              flexWrap: 'wrap',
            }}
          >
            <span><span style={{ color: 'var(--green)' }}>●</span> Model version</span>
            <span><span style={{ color: 'var(--red)' }}>●</span> Attack window</span>
            <span style={{ color: 'var(--amber)' }}>--- drift link</span>
          </div>
        </Panel>

        <Panel
          title={
            <span style={{ display: 'flex', gap: 0, flexWrap: 'wrap' }}>
              <button
                onClick={() => setActiveTab('alerts')}
                style={{
                  background: activeTab === 'alerts' ? 'var(--bg3)' : 'transparent',
                  border: 'none',
                  color: activeTab === 'alerts' ? 'var(--green)' : 'var(--text-dim)',
                  fontFamily: 'var(--mono)',
                  fontSize: 9,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                  padding: '0 10px 0 0',
                }}
              >
                Alert feed
              </button>
              <button
                onClick={() => setActiveTab('drift')}
                style={{
                  background: activeTab === 'drift' ? 'var(--bg3)' : 'transparent',
                  border: 'none',
                  color: activeTab === 'drift' ? 'var(--amber)' : 'var(--text-dim)',
                  fontFamily: 'var(--mono)',
                  fontSize: 9,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                  padding: '0 10px',
                }}
              >
                Drift log
              </button>
            </span>
          }
          badge={activeTab === 'alerts' ? `${alerts.length} events` : `${drifts.length} drifts`}
          badgeColor={activeTab === 'alerts' ? 'var(--green)' : 'var(--amber)'}
        >
          {activeTab === 'drift' && <AdaptationSummary drifts={drifts} stats={stats} />}
          <div style={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {activeTab === 'alerts' ? <AlertFeed alerts={alerts} /> : <DriftLog drifts={drifts} />}
          </div>
        </Panel>
      </div>
    </div>
  )
}
