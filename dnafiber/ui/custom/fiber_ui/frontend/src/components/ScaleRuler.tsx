import React from "react"

interface ScaleRulerProps {
  label: string
  barPx: number
  currentScale: number
}

export const ScaleRuler: React.FC<ScaleRulerProps> = ({
  label,
  barPx,
  currentScale,
}) => (
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
)
