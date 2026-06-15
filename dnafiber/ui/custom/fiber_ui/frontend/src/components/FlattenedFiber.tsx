import React from "react"
import { Fiber } from "../types"
import {
  parsePolylinePoints,
  calculatePolylineLength,
  formatLength,
  formatLengthPrecise,
} from "../utils"
import { normalizeColor } from "../utils/colors"

interface FlattenedFiberProps {
  fiber: Fiber
  pixelSize: number
  margin: number
  firstAnalogColor: string
  secondAnalogColor: string
}

export const FlattenedFiber: React.FC<FlattenedFiberProps> = ({
  fiber,
  pixelSize,
  margin,
  firstAnalogColor,
  secondAnalogColor,
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

  const firstLength = fiber.firstAnalogPx * pixelSize
  const secondLength = fiber.secondAnalogPx * pixelSize
  const totalLengthUm = firstLength + secondLength
  const ratio = fiber.ratio >= 0 ? fiber.ratio.toFixed(2) : "∞"
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
                {formatLength(fiber.segmentLengthsPx[i] * pixelSize)}
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
        <tspan fill={secondAnalogColor}>2nd</tspan>/
        <tspan fill={firstAnalogColor}>1st</tspan>: {ratio}
      </text>
    </g>
  )
}
