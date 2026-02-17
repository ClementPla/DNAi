import React, { useEffect, useRef, useState } from "react"

export interface ExportOptions {
  includeScaleRuler: boolean
  includeFlattenedFiber: boolean
  segmentationOffset: number
  bboxExpand: number
}

interface ContextMenuProps {
  x: number
  y: number
  fiberId: number
  onDownload: (options: ExportOptions) => void
  onClose: () => void
}

const Toggle: React.FC<{
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}> = ({ label, checked, onChange }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "5px 12px",
      color: "white",
      fontSize: 12,
      userSelect: "none",
      cursor: "pointer",
    }}
    onClick={() => onChange(!checked)}
  >
    <span style={{ opacity: checked ? 1 : 0.5 }}>{label}</span>
    <div
      style={{
        width: 28,
        height: 16,
        borderRadius: 8,
        background: checked ? "#4CAF50" : "rgba(255,255,255,0.2)",
        position: "relative",
        transition: "background 0.15s",
        flexShrink: 0,
        marginLeft: 12,
      }}
    >
      <div
        style={{
          width: 12,
          height: 12,
          borderRadius: 6,
          background: "white",
          position: "absolute",
          top: 2,
          left: checked ? 14 : 2,
          transition: "left 0.15s",
        }}
      />
    </div>
  </div>
)

export const ContextMenu: React.FC<ContextMenuProps> = ({
  x,
  y,
  fiberId,
  onDownload,
  onClose,
}) => {
  const menuRef = useRef<HTMLDivElement>(null)
  const [includeScaleRuler, setIncludeScaleRuler] = useState(true)
  const [includeFlattenedFiber, setIncludeFlattenedFiber] = useState(true)
  const [segmentationOffset, setSegmentationOffset] = useState(0)
  const [bboxExpand, setBboxExpand] = useState(1.0)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    const handleScroll = () => onClose()

    document.addEventListener("mousedown", handleClickOutside)
    document.addEventListener("scroll", handleScroll, true)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
      document.removeEventListener("scroll", handleScroll, true)
    }
  }, [onClose])

  return (
    <div
      ref={menuRef}
      style={{
        position: "fixed",
        left: x,
        top: y,
        zIndex: 9999,
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.15)",
        borderRadius: 6,
        boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
        padding: "4px 0",
        minWidth: 200,
        fontFamily: "system-ui, sans-serif",
        fontSize: 13,
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "4px 12px 2px",
          color: "rgba(255,255,255,0.45)",
          fontSize: 11,
          userSelect: "none",
        }}
      >
        Fiber #{fiberId}
      </div>

      {/* Separator */}
      <div
        style={{
          height: 1,
          background: "rgba(255,255,255,0.1)",
          margin: "4px 0",
        }}
      />

      {/* Toggles */}
      <Toggle
        label="Scale ruler"
        checked={includeScaleRuler}
        onChange={setIncludeScaleRuler}
      />
      <Toggle
        label="Flattened fiber"
        checked={includeFlattenedFiber}
        onChange={setIncludeFlattenedFiber}
      />

      {/* Offset slider */}
      <div style={{ padding: "5px 12px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            color: "white",
            fontSize: 12,
            marginBottom: 4,
          }}
        >
          <span style={{ opacity: segmentationOffset !== 0 ? 1 : 0.5 }}>
            Overlay offset
          </span>
          <span
            style={{
              fontSize: 11,
              color: "rgba(255,255,255,0.5)",
              minWidth: 28,
              textAlign: "right",
            }}
          >
            {segmentationOffset}px
          </span>
        </div>
        <input
          type="range"
          min={-20}
          max={20}
          step={1}
          value={segmentationOffset}
          onChange={(e) => setSegmentationOffset(Number(e.target.value))}
          style={{
            width: "100%",
            height: 4,
            accentColor: "#4CAF50",
            cursor: "pointer",
          }}
        />
      </div>

      {/* Bbox expand slider */}
      <div style={{ padding: "5px 12px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            color: "white",
            fontSize: 12,
            marginBottom: 4,
          }}
        >
          <span style={{ opacity: bboxExpand !== 1.0 ? 1 : 0.5 }}>
            Crop expand
          </span>
          <span
            style={{
              fontSize: 11,
              color: "rgba(255,255,255,0.5)",
              minWidth: 28,
              textAlign: "right",
            }}
          >
            {bboxExpand.toFixed(1)}×
          </span>
        </div>
        <input
          type="range"
          min={1.0}
          max={4.0}
          step={0.1}
          value={bboxExpand}
          onChange={(e) => setBboxExpand(Number(e.target.value))}
          style={{
            width: "100%",
            height: 4,
            accentColor: "#4CAF50",
            cursor: "pointer",
          }}
        />
      </div>

      {/* Separator */}
      <div
        style={{
          height: 1,
          background: "rgba(255,255,255,0.1)",
          margin: "4px 0",
        }}
      />

      {/* Download button */}
      <div
        onClick={() => {
          onDownload({
            includeScaleRuler,
            includeFlattenedFiber,
            segmentationOffset,
            bboxExpand,
          })
          onClose()
        }}
        style={{
          padding: "6px 12px",
          color: "white",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 8,
          lineHeight: 1,
        }}
        onMouseEnter={(e) =>
          (e.currentTarget.style.background = "rgba(255,255,255,0.1)")
        }
        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
      >
        <span>Download as PNG</span>
      </div>
    </div>
  )
}
