import React, { useState } from 'react'

const TreeComponent = ({ data }) => {
  const [expandedNodes, setExpandedNodes] = useState(new Set(['root']))
  const [hoveredNode, setHoveredNode] = useState(null)

  const toggleNode = (nodeId) => {
    const next = new Set(expandedNodes)
    next.has(nodeId) ? next.delete(nodeId) : next.add(nodeId)
    setExpandedNodes(next)
  }

  const getNodeColor = (depth) => {
    if (depth === 0) return '#00ffff' // root 青蓝
    if (depth === 1) return '#ffd700' // 一级 金黄
    if (depth === 2) return '#ffa500' // 二级 橙色
    return '#cccccc' // 其余灰
  }
  const renderNode = (node, depth = 0, parentId = 'root', index = 0) => {
    if (!node) return null
  
    const safeCategory = node.category || 'unnamed'
    // ✅ 唯一ID：包含父路径 + 索引
    const nodeId = `${parentId}-${safeCategory}-${depth}-${index}`
    const isExpanded = expandedNodes.has(nodeId)
    const color = getNodeColor(depth)
    const isHovered = hoveredNode === nodeId
    const hasAnyContent = (node.items?.length || 0) > 0 || (node.children?.length || 0) > 0
  
    return (
      <div
        key={nodeId}
        style={{
          marginLeft: depth * 20,
          borderLeft: depth > 0 ? '1px dashed #333' : 'none',
          paddingLeft: 6,
          transition: 'all 0.2s ease-out',
        }}
      >
        {/* 标题行 */}
        <div
          onClick={() => hasAnyContent && toggleNode(nodeId)}
          onMouseEnter={() => setHoveredNode(nodeId)}
          onMouseLeave={() => setHoveredNode(null)}
          style={{
            cursor: hasAnyContent ? 'pointer' : 'default',
            color: isHovered ? '#ffffff' : color,
            fontWeight: depth === 0 ? 'bold' : 'normal',
            userSelect: 'none',
          }}
        >
          {hasAnyContent ? (isExpanded ? '▼ ' : '▶ ') : '▷ '}
          [{safeCategory}]
        </div>
  
        {/* 展开后显示内容 */}
        {isExpanded && (
          <>
            {node.items?.map((item, i) => {
              // Handle newline characters by splitting and rendering each line
              const lines = String(item).split('\n')
              return (
                <div key={`${nodeId}-item-${i}`}>
                  {lines.map((line, lineIndex) => (
                    <div
                      key={`${nodeId}-item-${i}-line-${lineIndex}`}
                      style={{
                        marginLeft: 20,
                        color: '#aaffaa',
                        fontFamily: 'monospace',
                        lineHeight: '1.5em',
                        whiteSpace: 'pre-wrap',
                      }}
                    >
                      {lineIndex === 0 ? '• ' : '  '}{line}
                    </div>
                  ))}
                </div>
              )
            })}
  
            {node.children?.map((child, i) =>
              renderNode(child, depth + 1, nodeId, i) // ✅ 把 index 传递下去
            )}
          </>
        )}
      </div>
    )
  }
  
  const renderContent = () => {
    if (!data || Object.keys(data).length === 0) {
      return (
        <div
          style={{
            textAlign: 'center',
            color: '#777',
            padding: '40px',
            fontStyle: 'italic',
          }}
        >
          No data available. Waiting for stream updates...
        </div>
      )
    }

    if (typeof data === 'string') {
      try {
        const parsed = JSON.parse(data)
        return renderNode(parsed)
      } catch {
        return (
          <pre style={{ 
            color: '#aaffaa', 
            whiteSpace: 'pre-wrap',
            fontFamily: 'Consolas, Menlo, monospace'
          }}>
            {data}
          </pre>
        )
      }
    }

    if (Array.isArray(data)) {
      return data.map((n, i) => renderNode(n, 0, `root-${i}`))
    }

    return renderNode(data)
  }

  return (
    <div
      style={{
        color: '#fff',
        padding: '10px 20px',
        backgroundColor: '#0b0b0b',
        borderRadius: 8,
        fontFamily: 'Consolas, Menlo, monospace',
        overflowX: 'auto',
        whiteSpace: 'pre-wrap',
      }}
    >
      {renderContent()}
    </div>
  )
}

export default TreeComponent