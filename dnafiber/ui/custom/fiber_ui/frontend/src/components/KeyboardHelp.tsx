import React from "react"

const shortcuts = [
  { key: "T", label: "Toggle image" },
  { key: "N", label: "Next uninspected" },
  { key: "H", label: "Hide inspected" },
  { key: "E", label: "Mark as error" },
  { key: "I", label: "Mark inspected" },
  { key: "F", label: "Filter panel" },
]

export const KeyboardHelp: React.FC = () => (
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
    {shortcuts.map(({ key, label }) => (
      <div key={key}>
        <kbd>{key}</kbd> {label}
      </div>
    ))}
  </div>
)
