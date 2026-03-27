import React, { useEffect, useMemo, useRef } from 'react'
import * as d3 from 'd3'

function buildGraph(alerts, drifts) {
  const nodes = []
  const links = []

  const versions = [...new Set(alerts.map(alert => alert.model_version))]
  for (const version of versions) {
    const versionAlerts = alerts.filter(alert => alert.model_version === version)
    const attackCount = versionAlerts.filter(alert => alert.prediction === 1).length
    nodes.push({
      id: `v${version}`,
      version,
      label: `v${version}`,
      size: 10 + Math.min(versionAlerts.length * 0.25, 18),
      attacks: attackCount,
      total: versionAlerts.length,
      isDrift: drifts.some(drift => drift.model_version === version),
    })
  }

  for (let i = 1; i < nodes.length; i += 1) {
    links.push({ source: nodes[i - 1].id, target: nodes[i].id, drift: true })
  }

  const recent = alerts.filter(alert => alert.prediction === 1).slice(0, 30)
  const clusters = []
  for (let i = 0; i < Math.min(recent.length, 8); i += 1) {
    const alert = recent[i]
    clusters.push({
      id: `a${i}`,
      label: `W${alert.window_id}`,
      size: 4,
      isAlert: true,
      version: alert.model_version,
    })
    links.push({ source: `v${alert.model_version}`, target: `a${i}`, drift: false })
  }

  return { nodes: [...nodes, ...clusters], links }
}

export default function EvolutionGraph({ alerts, drifts }) {
  const svgRef = useRef(null)
  const { nodes, links } = useMemo(
    () => buildGraph(alerts.slice(0, 200), drifts),
    [alerts.length, drifts.length]
  )

  useEffect(() => {
    if (!svgRef.current || !nodes.length) return undefined

    const el = svgRef.current
    const width = el.clientWidth || 320
    const height = el.clientHeight || 220

    d3.select(el).selectAll('*').remove()

    const svg = d3.select(el).attr('width', width).attr('height', height)

    const simulation = d3
      .forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(68))
      .force('charge', d3.forceManyBody().strength(-135))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide(d => d.size + 8))

    const link = svg
      .append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => (d.drift ? 'var(--amber)' : 'var(--border2)'))
      .attr('stroke-width', d => (d.drift ? 1.5 : 0.8))
      .attr('stroke-dasharray', d => (d.drift ? '4 3' : 'none'))
      .attr('opacity', 0.72)

    const node = svg
      .append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer')

    node
      .append('circle')
      .attr('r', d => d.size)
      .attr('fill', d => {
        if (d.isAlert) return 'rgba(255,77,106,0.28)'
        if (d.version === 0) return 'var(--blue-dim)'
        return 'var(--green-dim)'
      })
      .attr('stroke', d => {
        if (d.isAlert) return 'var(--red)'
        if (d.version === 0) return 'var(--blue)'
        return 'var(--green)'
      })
      .attr('stroke-width', d => (d.isAlert ? 1 : 1.5))
      .attr('filter', d => (d.isAlert ? 'none' : 'drop-shadow(0 0 6px rgba(0,255,163,0.3))'))

    node
      .append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', d => d.size + 12)
      .attr('fill', d => (d.isAlert ? 'var(--red)' : 'var(--text-dim)'))
      .attr('font-family', 'var(--mono)')
      .attr('font-size', 9)

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    return () => simulation.stop()
  }, [nodes.length, links.length])

  if (!nodes.length) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: 'var(--text-dim)',
          fontSize: 11,
        }}
      >
        Awaiting model evolution data...
      </div>
    )
  }

  return <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
}
