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

import { Fiber } from "./types"
import { TOOLBAR_HEIGHT, VIEWER_HEIGHT, SIDEBAR_WIDTH } from "./utils/constants"
import { getScaleInfo } from "./utils"
import { useInspection, useFiberTypes, useFiberNavigation } from "./hooks/index"
import {
  FlattenedFiber,
  Toolbar,
  Sidebar,
  ScaleRuler,
  KeyboardHelp,
} from "./components"

const engine = new Styletron()

function FiberComponent(
  this: any,
  { args, disabled, width, theme }: ComponentProps
): ReactElement {
  let { image, elements, image_w, image_h, pixel_size, error_threshold } = args

  elements = elements.map((el: any): Fiber => {
    if (typeof el === "string") return JSON.parse(el) as Fiber
    return el as Fiber
  })

  // --- Display state ---
  const [showOnlyPolylines, setShowOnlyPolylines] = useState(false)
  const [hideBbox, setHideBbox] = useState(false)
  const [animated, setAnimated] = useState(false)
  const [strokeScale, setStrokeScale] = useState([1])
  const [currentScale, setCurrentScale] = useState(1)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // --- Hooks ---
  const inspection = useInspection()
  const fiberTypes = useFiberTypes(elements)

  // --- Refs ---
  const transformRef = useRef<ReactZoomPanPinchRef | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // --- Layout calculations ---
  const effectiveWidth = sidebarOpen ? width - SIDEBAR_WIDTH : width
  const svgWidth = effectiveWidth
  const svgHeight = Math.min(
    VIEWER_HEIGHT,
    effectiveWidth * (image_h / image_w)
  )
  const scaleX = effectiveWidth / image_w
  const scaleY = VIEWER_HEIGHT / image_h
  const fitScale = Math.min(scaleX, scaleY)
  const margin = (4 * Math.min(image_w, image_h)) / 1024
  const default_radius = Math.min(image_w, image_h) / 1024

  // --- Navigation ---
  const navigateToNextUninspected = useFiberNavigation({
    elements,
    inspectedFibers: inspection.inspectedFibers,
    setInspectedFibers: inspection.setInspectedFibers,
    hiddenTypes: fiberTypes.hiddenTypes,
    transformRef,
    effectiveWidth,
    svgWidth,
    svgHeight,
    image_w,
    image_h,
  })

  // --- Derived data ---
  const visibleElements = useMemo(() => {
    let filtered = elements.filter(
      (el: Fiber) => !fiberTypes.hiddenTypes.has(el.type)
    )
    if (inspection.hideInspected) {
      filtered = filtered.filter(
        (el: Fiber) =>
          !inspection.inspectedFibers.has(el.fiber_id) ||
          inspection.selectedFibers.includes(el.fiber_id)
      )
    }
    return filtered
  }, [
    elements,
    inspection.hideInspected,
    inspection.inspectedFibers,
    inspection.selectedFibers,
    fiberTypes.hiddenTypes,
  ])

  const visibleTypeElements = useMemo(
    () => elements.filter((el: Fiber) => !fiberTypes.hiddenTypes.has(el.type)),
    [elements, fiberTypes.hiddenTypes]
  )
  const inspectedCount = visibleTypeElements.filter((el: Fiber) =>
    inspection.inspectedFibers.has(el.fiber_id)
  ).length
  const totalCount = visibleTypeElements.length

  // --- Scale info ---
  const { barPx, label } = getScaleInfo(pixel_size, currentScale, fitScale)
  const themeMode = theme?.base === "dark" ? LightTheme : DarkTheme

  // --- Effects ---
  useEffect(() => {
    Streamlit.setFrameHeight(TOOLBAR_HEIGHT + VIEWER_HEIGHT + 32)
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => containerRef.current?.focus(), 100)
    return () => clearTimeout(timer)
  }, [])

  // --- Handlers ---
  const handleTransform = useCallback((ref: ReactZoomPanPinchRef) => {
    setCurrentScale(ref.state.scale)
  }, [])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      switch (e.key.toLowerCase()) {
        case "t":
          setShowOnlyPolylines((prev) => !prev)
          break
        case "n":
          navigateToNextUninspected()
          break
        case "h":
          inspection.setHideInspected((prev: boolean) => !prev)
          break
        case "e":
          if (inspection.hoveredFiberId !== null)
            inspection.handleFiberClick(inspection.hoveredFiberId)
          break
        case "i":
          if (inspection.hoveredFiberId !== null)
            inspection.markInspected(inspection.hoveredFiberId)
          break
        case "f":
          setSidebarOpen((prev) => !prev)
          break
      }
    },
    [navigateToNextUninspected, inspection]
  )

  return (
    <StyletronProvider value={engine}>
      <BaseProvider theme={themeMode}>
        <div
          ref={containerRef}
          tabIndex={0}
          onClick={() => containerRef.current?.focus()}
          onKeyDown={handleKeyDown}
          style={{ width: "100%", outline: "none" }}
        >
          <Toolbar
            showOnlyPolylines={showOnlyPolylines}
            onTogglePolylines={() => setShowOnlyPolylines((p) => !p)}
            hideBbox={hideBbox}
            onSetHideBbox={setHideBbox}
            animated={animated}
            onSetAnimated={setAnimated}
            strokeScale={strokeScale}
            onSetStrokeScale={setStrokeScale}
            hideInspected={inspection.hideInspected}
            onSetHideInspected={inspection.setHideInspected}
            inspectedCount={inspectedCount}
            totalCount={totalCount}
            onNavigateNext={navigateToNextUninspected}
            onRecenter={() => transformRef.current?.resetTransform()}
            onToggleSidebar={() => setSidebarOpen((p) => !p)}
            onResetAll={inspection.resetAll}
            onSend={() =>
              Streamlit.setComponentValue(inspection.selectedFibers)
            }
            selectedCount={inspection.selectedFibers.length}
            theme={theme}
          />

          <div
            style={{
              display: "flex",
              position: "relative",
              width: "100%",
              height: VIEWER_HEIGHT,
            }}
          >
            {/* Viewer */}
            <div
              style={{
                position: "relative",
                flex: 1,
                height: VIEWER_HEIGHT,
                backgroundColor: DarkTheme.colors.backgroundPrimary,
                borderRadius: 8,
                overflow: "hidden",
                transition: "flex 0.25s ease",
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
                      const isInspected = inspection.inspectedFibers.has(
                        el.fiber_id
                      )
                      const isSelected = inspection.selectedFibers.includes(
                        el.fiber_id
                      )
                      const bboxOpacity =
                        isInspected && !isSelected ? 0.85 : 1.0
                      const bboxStroke = isSelected
                        ? "blue"
                        : el.proba_error > error_threshold
                          ? "red"
                          : isInspected
                            ? "gray"
                            : "white"

                      return (
                        <g
                          key={idx}
                          className="rect-group"
                          opacity={
                            isInspected &&
                            !isSelected &&
                            !inspection.hideInspected
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
                              inspection.handleFiberMouseEnter(el.fiber_id)
                            }
                            onMouseLeave={inspection.handleFiberMouseLeave}
                            onClick={() =>
                              inspection.handleFiberClick(el.fiber_id)
                            }
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
                              strokeWidth={
                                default_radius * strokeScale[0] * 1.5
                              }
                              rx={default_radius}
                              pointerEvents="none"
                            />
                          )}

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

                    {inspection.hoveredFiberId !== null &&
                      (() => {
                        const hovered = elements.find(
                          (el: Fiber) =>
                            el.fiber_id === inspection.hoveredFiberId
                        )
                        return hovered ? (
                          <FlattenedFiber
                            fiber={hovered}
                            pixelSize={pixel_size}
                            margin={margin}
                          />
                        ) : null
                      })()}
                  </svg>
                </TransformComponent>
              </TransformWrapper>

              <ScaleRuler
                label={label}
                barPx={barPx}
                currentScale={currentScale}
              />
              <KeyboardHelp />
            </div>

            <Sidebar
              open={sidebarOpen}
              fiberTypes={fiberTypes.fiberTypes}
              typeCounts={fiberTypes.typeCounts}
              hiddenTypes={fiberTypes.hiddenTypes}
              onToggleType={fiberTypes.toggleType}
              onShowAll={fiberTypes.showAllTypes}
              onHideAll={fiberTypes.hideAllTypes}
              visibleCount={visibleElements.length}
              totalCount={elements.length}
            />
          </div>
        </div>
      </BaseProvider>
    </StyletronProvider>
  )
}

export default withStreamlitConnection(FiberComponent)
