import React from "react"
import { Fiber } from "../types"
import {
  parsePolylinePoints,
  calculatePolylineLength,
  formatLength,
  formatLengthPrecise,
  isGreen,
  isRed,
} from "../utils"

interface FlattenedFiberProps {
  fiber: Fiber
  pixelSize: number
  margin: number
}

export const FlattenedFiber: React.FC<FlattenedFiberProps> = ({
  fiber,
  pixelSize,
  margin,
}) => {
  const segments = fiber.points.map((line, i) => {
    const pts = parsePolylinePoints(line)
    const lengthPx = calculatePolylineLength(pts)
    const lengthUm = lengthPx * pixelSize
    return { pts, lengthPx, lengthUm, color: fiber.colors[i] }
  })

  const firstPt = segments[0]?.pts[0]
  const lastSeg = segments[segments.length - 1]
  const lastPt = lastSeg?.pts[lastSeg.pts.length - 1]

  const shouldReverse = firstPt && lastPt && firstPt.x > lastPt.x
  const orderedSegments = shouldReverse ? [...segments].reverse() : segments

  const totalLengthPx = orderedSegments.reduce((a, s) => a + s.lengthPx, 0)
  const totalLengthUm = orderedSegments.reduce((a, s) => a + s.lengthUm, 0)

  const greenLength = orderedSegments
    .filter((s) => isGreen(s.color))
    .reduce((a, s) => a + s.lengthUm, 0)
  const redLength = orderedSegments
    .filter((s) => isRed(s.color))
    .reduce((a, s) => a + s.lengthUm, 0)
  const ratio = redLength > 0 ? (greenLength / redLength).toFixed(2) : "∞"

  const barWidth = totalLengthPx
  const barHeight = 8
  const fontSize = 8

  const barX = fiber.x + fiber.width / 2 - barWidth / 2
  const barY = fiber.y + fiber.height + margin * 3

  let currentX = barX

  return (
    <g className="fiber-flatten" pointerEvents="none">
      <rect
        x={barX - 4}
        y={barY - 4}
        width={barWidth + 8}
        height={barHeight + 22}
        fill="rgba(0,0,0,0.7)"
        rx={4}
      />

      {orderedSegments.map((seg, i) => {
        const segWidth = seg.lengthPx
        const x = currentX
        currentX += segWidth

        return (
          <g key={i}>
            <rect
              x={x}
              y={barY}
              width={segWidth}
              height={barHeight}
              fill={seg.color}
              stroke="rgba(0,0,0,0.3)"
              strokeWidth={1}
            />
            {segWidth > 15 && (
              <text
                x={x + segWidth / 2}
                y={barY + barHeight / 2}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="white"
                fontSize={fontSize}
                fontFamily="system-ui, sans-serif"
                style={{ textShadow: "0 0 2px black, 0 0 2px black" }}
              >
                {formatLength(seg.lengthUm)}
              </text>
            )}
          </g>
        )
      })}

      <text
        x={barX + barWidth / 2}
        y={barY + barHeight + 12}
        textAnchor="middle"
        fill="white"
        fontSize={fontSize + 1}
        fontFamily="system-ui, sans-serif"
      >
        Total: {formatLengthPrecise(totalLengthUm)}
        {" · "}
        <tspan fill="#90EE90">G</tspan>/<tspan fill="#FF6B6B">R</tspan>: {ratio}
      </text>
    </g>
  )
}
