import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"

import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  ReactElement,
  useRef,
} from "react"

import "./style.css"

import { Provider as StyletronProvider } from "styletron-react"
import { Client as Styletron } from "styletron-engine-atomic"
import { BaseProvider, LightTheme, DarkTheme } from "baseui"

import {
  TransformWrapper,
  TransformComponent,
  ReactZoomPanPinchRef,
} from "react-zoom-pan-pinch"
import { Button } from "baseui/button"
import { Switch } from "@base-ui-components/react/switch"
import { Slider } from "baseui/slider"
import switch_styles from "./MySwitch.module.css"

interface Fiber {
  width: number
  height: number
  x: number
  y: number
  fiber_id: number
  ratio: number
  type: string
  points: string[]
  colors: string[]
  is_error: boolean
  pixel_size: number
}

function parsePolylinePoints(pointsStr: string): { x: number; y: number }[] {
  return pointsStr
    .split(" ")
    .filter(Boolean)
    .map((p) => {
      const [x, y] = p.split(",").map(Number)
      return { x, y }
    })
}

function calculatePolylineLength(points: { x: number; y: number }[]): number {
  let length = 0
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x
    const dy = points[i].y - points[i - 1].y
    length += Math.sqrt(dx * dx + dy * dy)
  }
  return length
}

interface FlattenedFiberProps {
  fiber: Fiber
  pixelSize: number
  margin: number
}

const FlattenedFiber: React.FC<FlattenedFiberProps> = ({
  fiber,
  pixelSize,
  margin,
}) => {
  // Parse all segments with their data
  const segments = fiber.points.map((line, i) => {
    const pts = parsePolylinePoints(line)
    const lengthPx = calculatePolylineLength(pts)
    const lengthUm = lengthPx * pixelSize
    return { pts, lengthPx, lengthUm, color: fiber.colors[i] }
  })

  // Determine fiber orientation: compare x of first point vs last point
  const firstPt = segments[0]?.pts[0]
  const lastSeg = segments[segments.length - 1]
  const lastPt = lastSeg?.pts[lastSeg.pts.length - 1]

  // If fiber goes right-to-left, reverse the segment order
  const shouldReverse = firstPt && lastPt && firstPt.x > lastPt.x
  const orderedSegments = shouldReverse ? [...segments].reverse() : segments

  const totalLengthPx = orderedSegments.reduce((a, s) => a + s.lengthPx, 0)
  const totalLengthUm = orderedSegments.reduce((a, s) => a + s.lengthUm, 0)
  // Calculate green/red ratio
  const isGreen = (color: string) =>
    color.toLowerCase().includes("green") ||
    color === "#00ff00" ||
    color === "#00FF00"
  const isRed = (color: string) =>
    color.toLowerCase().includes("red") ||
    color === "#ff0000" ||
    color === "#f00" ||
    color === "#FF0000"

  const greenLength = orderedSegments
    .filter((s) => isGreen(s.color))
    .reduce((a, s) => a + s.lengthUm, 0)
  const redLength = orderedSegments
    .filter((s) => isRed(s.color))
    .reduce((a, s) => a + s.lengthUm, 0)
  const ratio = redLength > 0 ? (greenLength / redLength).toFixed(2) : "∞"

  // Use actual fiber length as bar width
  const barWidth = totalLengthPx
  const barHeight = 8
  const fontSize = 8

  // Center the bar under the fiber's bounding box
  const barX = fiber.x + fiber.width / 2 - barWidth / 2
  const barY = fiber.y + fiber.height + margin * 3

  let currentX = barX

  return (
    <g className="fiber-flatten" pointerEvents="none">
      {/* Background */}
      <rect
        x={barX - 4}
        y={barY - 4}
        width={barWidth + 8}
        height={barHeight + 22}
        fill="rgba(0,0,0,0.7)"
        rx={4}
      />

      {/* Colored segments */}
      {orderedSegments.map((seg, i) => {
        const segWidth = seg.lengthPx
        const x = currentX
        currentX += segWidth

        const lengthLabel =
          seg.lengthUm >= 1000
            ? `${(seg.lengthUm / 1000).toFixed(1)}mm`
            : `${seg.lengthUm.toFixed(1)}μm`

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
                {lengthLabel}
              </text>
            )}
          </g>
        )
      })}

      {/* Total length and ratio label */}
      <text
        x={barX + barWidth / 2}
        y={barY + barHeight + 12}
        textAnchor="middle"
        fill="white"
        fontSize={fontSize + 1}
        fontFamily="system-ui, sans-serif"
      >
        Total:{" "}
        {totalLengthUm >= 1000
          ? `${(totalLengthUm / 1000).toFixed(2)} mm`
          : `${totalLengthUm.toFixed(1)} μm`}
        {" · "}
        <tspan fill="#90EE90">G</tspan>/<tspan fill="#FF6B6B">R</tspan>: {ratio}
      </text>
    </g>
  )
}

const engine = new Styletron()

function getScaleInfo(
  pixelSize: number, // um per image pixel
  currentScale: number, // multiplier from react-zoom-pan-pinch
  baseScale: number // screen pixels per viewBox unit at zoom 1.0
) {
  const actualScreenPixelsPerUnit = baseScale * currentScale
  const micronsPerScreenPixel = pixelSize / actualScreenPixelsPerUnit

  const barPx = 100
  const totalUm = barPx * micronsPerScreenPixel

  let label: string
  if (totalUm >= 1000) {
    label = `${(totalUm / 1000).toPrecision(3)} mm`
  } else {
    label = `${totalUm.toPrecision(3)} μm`
  }

  return { barPx, label }
}

function FiberComponent(
  this: any,
  { args, disabled, width, theme }: ComponentProps
): ReactElement {
  let { image, elements, image_w, image_h, pixel_size } = args
  const [showOnlyPolylines, setShowOnlyPolylines] = useState(false)

  elements = elements.map((el: any): Fiber => {
    if (typeof el === "string") return JSON.parse(el) as Fiber
    return el as Fiber
  })

  const [hideBbox, setHideBbox] = useState(false)
  const [animated, setAnimated] = useState(false)
  const [selectedFibers, setSelectedFibers] = useState<number[]>([])
  const [hoveredFiberId, setHoveredFiberId] = useState<number | null>(null)
  const transformRef = useRef<ReactZoomPanPinchRef | null>(null)

  // Fixed heights in pixels
  const TOOLBAR_HEIGHT = 56
  const VIEWER_HEIGHT = 650

  // Tell Streamlit the exact height
  useEffect(() => {
    Streamlit.setFrameHeight(TOOLBAR_HEIGHT + VIEWER_HEIGHT + 32)
  }, [])
  // SVG fills width, height based on aspect ratio but capped
  const svgWidth = width
  const svgHeight = Math.min(VIEWER_HEIGHT, width * (image_h / image_w))
  const scaleX = width / image_w
  const scaleY = VIEWER_HEIGHT / image_h
  const fitScale = Math.min(scaleX, scaleY)

  const margin = (4 * Math.min(image_w, image_h)) / 1024
  const default_radius = Math.min(image_w, image_h) / 1024

  const handleToggle = () => setShowOnlyPolylines((prev) => !prev)
  const handleRecenter = () => transformRef.current?.resetTransform()

  const [currentScale, setCurrentScale] = useState(1)

  const handleTransform = useCallback((ref: ReactZoomPanPinchRef) => {
    setCurrentScale(ref.state.scale)
  }, [])

  const [strokeScale, setstrokeScaleValue] = useState([1])
  const themeMode = theme?.base === "dark" ? LightTheme : DarkTheme

  // Scale bar info
  const { barPx, label } = getScaleInfo(pixel_size, currentScale, fitScale)
  const fontSize = "12px"
  const buttonOverrides = {
    BaseButton: {
      style: {
        fontSize: fontSize,
        paddingTop: "4px",
        paddingBottom: "4px",
        paddingLeft: "8px",
        paddingRight: "8px",
      },
    },
  }

  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={themeMode}>
        <div style={{ width: "100%" }}>
          {/* Toolbar */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              height: TOOLBAR_HEIGHT,
              overflowX: "auto",
              overflowY: "hidden",
              whiteSpace: "nowrap",
              // @ts-ignore
              "--color-gray-100":
                (theme as any)?.colors?.backgroundTertiary ?? "#f0f0f0",
              "--color-gray-200":
                (theme as any)?.colors?.borderOpaque ?? "#e0e0e0",
              "--color-gray-500":
                (theme as any)?.colors?.borderSelected ?? "#b0b0b0",
              "--color-gray-700":
                (theme as any)?.colors?.contentPrimary ?? "#333333",
              "--color-blue": (theme as any)?.colors?.accent ?? "#276ef1",
            }}
          >
            <Button
              size="compact"
              overrides={buttonOverrides}
              onClick={handleToggle}
            >
              Toggle (T)
            </Button>
            <Button
              size="compact"
              overrides={buttonOverrides}
              onClick={handleRecenter}
            >
              Recenter
            </Button>

            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Switch.Root
                checked={hideBbox}
                onCheckedChange={setHideBbox}
                className={switch_styles.Switch}
              >
                <Switch.Thumb className={switch_styles.Thumb} />
              </Switch.Root>
              <span style={{ fontSize: fontSize }}>Hide BBoxes</span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <Switch.Root
                checked={animated}
                onCheckedChange={setAnimated}
                className={switch_styles.Switch}
              >
                <Switch.Thumb className={switch_styles.Thumb} />
              </Switch.Root>
              <span style={{ fontSize: fontSize }}>Animated</span>
            </div>

            <span className="slider-container">
              <Slider
                value={strokeScale}
                max={5}
                min={0.1}
                step={0.1}
                persistentThumb
                onChange={({ value }) => value && setstrokeScaleValue(value)}
                overrides={{ TickBar: { style: () => ({ opacity: 0 }) } }}
              />
            </span>

            <span style={{ fontSize: fontSize }}>{elements.length} fibers</span>

            <Button
              size="compact"
              overrides={buttonOverrides}
              onClick={() => setSelectedFibers([])}
            >
              Clear
            </Button>
            <Button
              size="compact"
              overrides={buttonOverrides}
              onClick={() => Streamlit.setComponentValue(selectedFibers)}
            >
              Send ({selectedFibers.length})
            </Button>
          </div>

          {/* Viewer container */}
          <div
            style={{
              position: "relative",
              width: "100%",
              height: VIEWER_HEIGHT,
              backgroundColor: DarkTheme.colors.backgroundPrimary,
              borderRadius: 8,
              overflow: "hidden",
            }}
          >
            <TransformWrapper
              ref={transformRef}
              disabled={disabled}
              minScale={0.75 * Math.max(fitScale, 0.9)}
              maxScale={20}
              initialScale={Math.max(fitScale, 0.9)}
              centerOnInit={true}
              wheel={{ smoothStep: 0.01, step: 0.5 }}
              onTransformed={handleTransform}
              onInit={(ref) => setCurrentScale(ref.state.scale)}
            >
              <TransformComponent
                wrapperStyle={{ width: "100%", height: "100%" }}
                contentStyle={{
                  width: "100%",
                  height: "100%",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <svg
                  width={svgWidth}
                  height={svgHeight}
                  viewBox={`0 0 ${image_w} ${image_h}`}
                  preserveAspectRatio="xMidYMid meet"
                  style={{
                    backgroundColor: DarkTheme.colors.backgroundPrimary,
                  }}
                >
                  <image
                    width={image_w}
                    height={image_h}
                    href={image}
                    className={`image ${showOnlyPolylines ? "hidden" : ""}`}
                  />

                  {elements.map((el: Fiber, idx: number) => (
                    <g key={idx} className="rect-group">
                      <rect
                        x={el.x - margin}
                        y={el.y - margin}
                        width={el.width + margin * 2}
                        height={el.height + margin * 2}
                        fill="none"
                        stroke={el.is_error ? "red" : "white"}
                        strokeWidth={default_radius * strokeScale[0]}
                        className={`hover-target ${showOnlyPolylines || hideBbox ? "hidden" : ""}`}
                        rx={default_radius}
                        onMouseEnter={() => setHoveredFiberId(el.fiber_id)}
                        onMouseLeave={() => setHoveredFiberId(null)}
                        onClick={() => {
                          setSelectedFibers((prev) =>
                            prev.includes(el.fiber_id)
                              ? prev.filter((i) => i !== el.fiber_id)
                              : [...prev, el.fiber_id]
                          )
                        }}
                      >
                        <title>
                          Fiber id: {el.fiber_id.toFixed(0)}, Ratio:{" "}
                          {el.ratio.toFixed(2)}
                        </title>
                      </rect>

                      {selectedFibers.includes(el.fiber_id) && (
                        <rect
                          x={el.x - margin}
                          y={el.y - margin}
                          width={el.width + margin * 2}
                          height={el.height + margin * 2}
                          fill="none"
                          stroke="blue"
                          strokeWidth={default_radius * strokeScale[0] * 1.5}
                          rx={default_radius}
                          pointerEvents="none"
                        />
                      )}

                      <g className="hover-paths">
                        {el.points.map((line: string, line_idx: number) => (
                          <polyline
                            className={animated ? "fibers-animated" : "fibers"}
                            key={`${line_idx}_${idx}`}
                            points={line}
                            fill="none"
                            stroke={el.colors[line_idx]}
                            strokeWidth={default_radius * strokeScale[0]}
                            opacity={showOnlyPolylines || hideBbox ? 1.0 : 0.0}
                          />
                        ))}
                      </g>
                    </g>
                  ))}
                  {hoveredFiberId !== null &&
                    (() => {
                      const hoveredFiber = elements.find(
                        (el: Fiber) => el.fiber_id === hoveredFiberId
                      )
                      return hoveredFiber ? (
                        <FlattenedFiber
                          fiber={hoveredFiber}
                          pixelSize={pixel_size}
                          margin={margin}
                        />
                      ) : null
                    })()}
                </svg>
              </TransformComponent>
            </TransformWrapper>

            {/* Scale ruler - absolute positioned, fixed size */}
            <div
              style={{
                position: "absolute",
                bottom: 16,
                left: 16,
                background: "rgba(0,0,0,0.8)",
                padding: 10,
                borderRadius: 6,
                pointerEvents: "none",
              }}
            >
              <div
                style={{
                  color: "white",
                  fontSize: 12,
                  marginBottom: 4,
                  fontFamily: "system-ui",
                }}
              >
                {label}{" "}
                <span style={{ opacity: 0.5, marginLeft: 8 }}>
                  {currentScale.toFixed(1)}×
                </span>
              </div>
              <div
                style={{
                  width: barPx,
                  height: 4,
                  background: "white",
                  borderRadius: 2,
                  position: "relative",
                }}
              >
                {/* Left tick */}
                <div
                  style={{
                    position: "absolute",
                    left: 0,
                    top: -4,
                    width: 2,
                    height: 12,
                    background: "white",
                  }}
                />
                {/* Right tick */}
                <div
                  style={{
                    position: "absolute",
                    right: 0,
                    top: -4,
                    width: 2,
                    height: 12,
                    background: "white",
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </BaseProvider>
    </StyletronProvider>
  )
}

export default withStreamlitConnection(FiberComponent)
