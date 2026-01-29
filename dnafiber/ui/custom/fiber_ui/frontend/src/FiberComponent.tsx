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
  pixelSize: number,
  currentScale: number,
  baseScale: number
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

// Inspection delay in ms
const INSPECTION_DELAY = 600

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

  // New: inspection tracking
  const [inspectedFibers, setInspectedFibers] = useState<Set<number>>(new Set())
  const [hideInspected, setHideInspected] = useState(false)
  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Fixed heights in pixels
  const TOOLBAR_HEIGHT = 56
  const VIEWER_HEIGHT = 650

  useEffect(() => {
    Streamlit.setFrameHeight(TOOLBAR_HEIGHT + VIEWER_HEIGHT + 32)
  }, [])

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

  // Ref for the focusable container
  const containerRef = useRef<HTMLDivElement>(null)

  // Focus the container on mount and when clicking anywhere in it
  useEffect(() => {
    // Small delay to ensure iframe is ready
    const timer = setTimeout(() => {
      containerRef.current?.focus()
    }, 100)
    return () => clearTimeout(timer)
  }, [])

  const handleContainerClick = useCallback(() => {
    containerRef.current?.focus()
  }, [])

  // Mark fiber as inspected after hover delay
  const handleFiberMouseEnter = useCallback((fiberId: number) => {
    setHoveredFiberId(fiberId)

    // Start timer to mark as inspected
    hoverTimerRef.current = setTimeout(() => {
      setInspectedFibers((prev) => new Set(prev).add(fiberId))
    }, INSPECTION_DELAY)
  }, [])

  const handleFiberMouseLeave = useCallback(() => {
    setHoveredFiberId(null)

    // Cancel pending inspection
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current)
      hoverTimerRef.current = null
    }
  }, [])

  // Click also marks as inspected immediately
  const handleFiberClick = useCallback((fiberId: number) => {
    setInspectedFibers((prev) => new Set(prev).add(fiberId))
    setSelectedFibers((prev) =>
      prev.includes(fiberId)
        ? prev.filter((i) => i !== fiberId)
        : [...prev, fiberId]
    )
  }, [])
  const navigateToNextUninspected = useCallback(() => {
    const uninspected = elements.find(
      (el: Fiber) => !inspectedFibers.has(el.fiber_id)
    )
    if (!uninspected || !transformRef.current) return

    inspectedFibers.add(uninspected.fiber_id)
    setInspectedFibers(new Set(inspectedFibers))

    const fiberVBX = uninspected.x + uninspected.width / 2
    const fiberVBY = uninspected.y + uninspected.height / 2

    const scaleX = svgWidth / image_w
    const scaleY = svgHeight / image_h
    const actualScale = Math.min(scaleX, scaleY)

    const renderedWidth = image_w * actualScale
    const renderedHeight = image_h * actualScale

    const svgInternalOffsetX = (svgWidth - renderedWidth) / 2
    const svgInternalOffsetY = (svgHeight - renderedHeight) / 2

    const fiberInSvgX = svgInternalOffsetX + fiberVBX * actualScale
    const fiberInSvgY = svgInternalOffsetY + fiberVBY * actualScale

    const contentOffsetX = (width - svgWidth) / 2
    const contentOffsetY = (VIEWER_HEIGHT - svgHeight) / 2

    const fiberContentX = contentOffsetX + fiberInSvgX
    const fiberContentY = contentOffsetY + fiberInSvgY

    const targetScale = 10
    const vpCenterX = width / 2
    const vpCenterY = VIEWER_HEIGHT / 2

    const newX = vpCenterX - fiberContentX * targetScale
    const newY = vpCenterY - fiberContentY * targetScale

    transformRef.current.setTransform(newX, newY, targetScale, 300)
  }, [elements, inspectedFibers, width, svgWidth, svgHeight, image_w, image_h])

  // Filter elements based on hideInspected
  const visibleElements = useMemo(() => {
    if (!hideInspected) return elements
    return elements.filter(
      (el: Fiber) =>
        !inspectedFibers.has(el.fiber_id) ||
        selectedFibers.includes(el.fiber_id)
    )
  }, [elements, hideInspected, inspectedFibers, selectedFibers])

  // Progress stats
  const inspectedCount = inspectedFibers.size
  const totalCount = elements.length
  const progressPercent =
    totalCount > 0 ? (inspectedCount / totalCount) * 100 : 0

  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={themeMode}>
        <div
          ref={containerRef}
          tabIndex={0}
          onClick={handleContainerClick}
          onKeyDown={(e) => {
            // Handle keyboard events here instead of window listener
            if (e.target instanceof HTMLInputElement) return

            switch (e.key.toLowerCase()) {
              case "t":
                setShowOnlyPolylines((prev) => !prev)
                break
              case "n":
                navigateToNextUninspected()
                break
              case "h":
                setHideInspected((prev) => !prev)
                break
              case "e":
                if (hoveredFiberId !== null) {
                  handleFiberClick(hoveredFiberId)
                }
                break
              case "i":
                if (hoveredFiberId !== null) {
                  setInspectedFibers((prev) =>
                    new Set(prev).add(hoveredFiberId)
                  )
                }
                break
            }
          }}
          style={{
            width: "100%",
            outline: "none",
          }}
        >
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

            {/* Inspection controls */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 4,
                borderLeft: "1px solid rgba(255,255,255,0.2)",
                paddingLeft: 8,
                marginLeft: 4,
              }}
            >
              <Button
                size="compact"
                overrides={buttonOverrides}
                onClick={navigateToNextUninspected}
                disabled={inspectedCount >= totalCount}
              >
                Next (N)
              </Button>

              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <Switch.Root
                  checked={hideInspected}
                  onCheckedChange={setHideInspected}
                  className={switch_styles.Switch}
                >
                  <Switch.Thumb className={switch_styles.Thumb} />
                </Switch.Root>
                <span style={{ fontSize: fontSize }}>Hide inspected (H)</span>
              </div>

              {/* Progress indicator */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 2,
                  minWidth: 80,
                }}
              >
                <span style={{ fontSize: "10px", opacity: 0.8 }}>
                  {inspectedCount}/{totalCount} inspected
                </span>
                <div
                  style={{
                    width: "100%",
                    height: 4,
                    backgroundColor: "rgba(255,255,255,0.2)",
                    borderRadius: 2,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${progressPercent}%`,
                      height: "100%",
                      backgroundColor: "#4CAF50",
                      transition: "width 0.2s ease",
                    }}
                  />
                </div>
              </div>
            </div>

            <div
              style={{
                borderLeft: "1px solid rgba(255,255,255,0.2)",
                paddingLeft: 8,
                marginLeft: 4,
                display: "flex",
                gap: 4,
              }}
            >
              <Button
                size="compact"
                overrides={buttonOverrides}
                onClick={() => {
                  setSelectedFibers([])
                  setInspectedFibers(new Set())
                }}
              >
                Reset All
              </Button>
              <Button
                size="compact"
                overrides={buttonOverrides}
                onClick={() => Streamlit.setComponentValue(selectedFibers)}
              >
                Send ({selectedFibers.length})
              </Button>
            </div>
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
              limitToBounds={false}
              centerZoomedOut={false}
              alignmentAnimation={{ disabled: true }}
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
                    // Override problematic CSS:
                    margin: 0,
                    minHeight: "unset",
                    maxHeight: "unset",
                    aspectRatio: "unset",
                    border: "none",
                  }}
                >
                  <image
                    width={image_w}
                    height={image_h}
                    href={image}
                    className={`image ${showOnlyPolylines ? "hidden" : ""}`}
                  />

                  {visibleElements.map((el: Fiber, idx: number) => {
                    const isInspected = inspectedFibers.has(el.fiber_id)
                    const isSelected = selectedFibers.includes(el.fiber_id)

                    // Visual distinction for inspection state
                    const bboxOpacity = isInspected && !isSelected ? 0.85 : 1.0
                    const bboxStroke = isSelected
                      ? "blue"
                      : el.is_error
                        ? "red"
                        : isInspected
                          ? "gray"
                          : "white"

                    return (
                      <g
                        key={idx}
                        className="rect-group"
                        opacity={
                          isInspected && !isSelected && !hideInspected
                            ? 0.95
                            : 1
                        }
                      >
                        <rect
                          x={el.x - margin}
                          y={el.y - margin}
                          width={el.width + margin * 2}
                          height={el.height + margin * 2}
                          fill="none"
                          stroke={bboxStroke}
                          strokeWidth={default_radius * strokeScale[0]}
                          opacity={bboxOpacity}
                          className={`hover-target ${showOnlyPolylines || hideBbox ? "hidden" : ""}`}
                          rx={default_radius}
                          onMouseEnter={() =>
                            handleFiberMouseEnter(el.fiber_id)
                          }
                          onMouseLeave={handleFiberMouseLeave}
                          onClick={() => handleFiberClick(el.fiber_id)}
                        >
                          <title>
                            Fiber id: {el.fiber_id.toFixed(0)}, Ratio:{" "}
                            {el.ratio.toFixed(2)}
                            {isInspected ? " ✓" : ""}
                          </title>
                        </rect>

                        {isSelected && (
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

                        {/* Small checkmark for inspected fibers */}
                        {isInspected && !hideBbox && !showOnlyPolylines && (
                          <circle
                            cx={el.x + el.width + margin}
                            cy={el.y - margin}
                            r={default_radius * 3}
                            fill={isSelected ? "blue" : "#4CAF50"}
                            pointerEvents="none"
                          />
                        )}

                        <g className="hover-paths">
                          {el.points.map((line: string, line_idx: number) => (
                            <React.Fragment key={`${line_idx}_${idx}`}>
                              <polyline
                                className={
                                  animated ? "fibers-animated" : "fibers"
                                }
                                points={line}
                                fill="none"
                                stroke="white"
                                strokeWidth={
                                  default_radius * strokeScale[0] * 1.8
                                }
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                opacity={
                                  showOnlyPolylines || hideBbox ? 1.0 : 0.0
                                }
                              />
                              <polyline
                                className={
                                  animated ? "fibers-animated" : "fibers"
                                }
                                points={line}
                                fill="none"
                                stroke={el.colors[line_idx]}
                                strokeWidth={default_radius * strokeScale[0]}
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                opacity={
                                  showOnlyPolylines || hideBbox ? 1.0 : 0.0
                                }
                              />
                            </React.Fragment>
                          ))}
                        </g>
                      </g>
                    )
                  })}

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

            {/* Scale ruler */}
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

            {/* Keyboard shortcuts help */}
            <div
              style={{
                position: "absolute",
                bottom: 16,
                right: 16,
                background: "rgba(0,0,0,0.8)",
                padding: 8,
                borderRadius: 6,
                fontSize: 10,
                color: "rgba(255,255,255,0.7)",
                fontFamily: "system-ui",
              }}
            >
              <div>
                <kbd>T</kbd> Toggle image
              </div>
              <div>
                <kbd>N</kbd> Next uninspected
              </div>
              <div>
                <kbd>H</kbd> Hide inspected
              </div>
              <div>
                <kbd>E</kbd> Mark as error
              </div>
              <div>
                <kbd>I</kbd> Mark inspected
              </div>
            </div>
          </div>
        </div>
      </BaseProvider>
    </StyletronProvider>
  )
}

export default withStreamlitConnection(FiberComponent)
