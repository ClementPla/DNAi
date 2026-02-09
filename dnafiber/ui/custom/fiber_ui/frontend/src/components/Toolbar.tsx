import React from "react"
import { Switch } from "@base-ui-components/react/switch"
import { Slider } from "baseui/slider"
import switch_styles from "../MySwitch.module.css"
import { TOOLBAR_HEIGHT } from "../utils"

/* ------------------------------------------------------------------ */
/*  Inline SVG icons                                                   */
/* ------------------------------------------------------------------ */

const Icon: React.FC<{ d: string; size?: number }> = ({ d, size = 13 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{
      border: "none",
      outline: "none",
      margin: 0,
      padding: 0,
      minHeight: "unset",
      maxHeight: "unset",
      aspectRatio: "unset",
      display: "block",
      flexShrink: 0,
    }}
  >
    <path d={d} />
  </svg>
)

const Icons = {
  layers: "M12 2 L2 7 L12 12 L22 7 Z M2 17 L12 22 L22 17 M2 12 L12 17 L22 12",
  crosshair:
    "M12 2 V6 M12 18 V22 M2 12 H6 M18 12 H22 M12 8 A4 4 0 1 0 12 16 A4 4 0 1 0 12 8",
  skipForward: "M5 4 L15 12 L5 20 Z M19 5 V19",
  filter: "M22 3 H2 L10 12.46 V19 L14 21 V12.46 Z",
  send: "M22 2 L11 13 M22 2 L15 22 L11 13 L2 9 Z",
  rotateCcw: "M1 4 V10 H7 M3.51 15 A9 9 0 1 0 5.64 5.64 L1 10",
} as const

/* ------------------------------------------------------------------ */
/*  Shared constants                                                   */
/* ------------------------------------------------------------------ */

const FONT =
  "'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace"

/* ------------------------------------------------------------------ */
/*  ToolBtn                                                            */
/* ------------------------------------------------------------------ */

const btnBase: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 5,
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
  fontFamily: FONT,
  fontSize: 11,
  fontWeight: 500,
  letterSpacing: "0.01em",
  lineHeight: 1,
  transition: "all 0.15s ease",
  whiteSpace: "nowrap",
  flexShrink: 0,
  height: 28,
  padding: "0 10px",
}

interface ToolBtnProps {
  icon?: string
  label?: string
  kbd?: string
  onClick: () => void
  disabled?: boolean
  accent?: boolean
  title?: string
}

const ToolBtn: React.FC<ToolBtnProps> = ({
  icon,
  label,
  kbd,
  onClick,
  disabled,
  accent,
  title,
}) => {
  const [hovered, setHovered] = React.useState(false)

  const bg = accent ? "rgba(99,162,255,0.12)" : "rgba(255,255,255,0.05)"
  const bgHover = accent ? "rgba(99,162,255,0.25)" : "rgba(255,255,255,0.12)"
  const fg = accent ? "#7EB8FF" : "rgba(255,255,255,0.72)"
  const fgHover = accent ? "#9ECBFF" : "#fff"

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...btnBase,
        background: hovered && !disabled ? bgHover : bg,
        color: hovered && !disabled ? fgHover : fg,
        ...(disabled ? { opacity: 0.3, cursor: "default" } : {}),
      }}
    >
      {icon && <Icon d={icon} />}
      {label && <span>{label}</span>}
      {kbd && (
        <kbd
          style={{
            fontSize: 9,
            opacity: 0.35,
            fontFamily: "inherit",
          }}
        >
          {kbd}
        </kbd>
      )}
    </button>
  )
}

/* ------------------------------------------------------------------ */
/*  SwitchRow                                                          */
/* ------------------------------------------------------------------ */

const SwitchRow: React.FC<{
  checked: boolean
  onChange: (v: boolean) => void
  label: string
}> = ({ checked, onChange, label }) => (
  <label
    style={{
      display: "flex",
      alignItems: "center",
      gap: 6,
      cursor: "pointer",
      fontSize: 11,
      color: checked ? "rgba(255,255,255,0.85)" : "rgba(255,255,255,0.4)",
      fontFamily: FONT,
      fontWeight: 500,
      transition: "color 0.15s",
      whiteSpace: "nowrap",
      flexShrink: 0,
      padding: "0 4px",
    }}
  >
    <Switch.Root
      checked={checked}
      onCheckedChange={onChange}
      className={switch_styles.Switch}
    >
      <Switch.Thumb className={switch_styles.Thumb} />
    </Switch.Root>
    {label}
  </label>
)

/* ------------------------------------------------------------------ */
/*  GroupPill — labeled cluster of controls                            */
/* ------------------------------------------------------------------ */

const GroupPill: React.FC<{
  children: React.ReactNode
  label?: string
}> = ({ children, label }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      gap: 0,
      flexShrink: 0,
    }}
  >
    {label && (
      <span
        style={{
          fontSize: 8,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "rgba(255,255,255,0.22)",
          fontFamily: FONT,
          paddingLeft: 6,
          marginBottom: 2,
          lineHeight: 1,
        }}
      >
        {label}
      </span>
    )}
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 8,
        padding: "3px 5px",
      }}
    >
      {children}
    </div>
  </div>
)

/* ------------------------------------------------------------------ */
/*  Circular progress ring                                            */
/* ------------------------------------------------------------------ */

const ProgressRing: React.FC<{
  inspected: number
  total: number
}> = ({ inspected, total }) => {
  const pct = total > 0 ? (inspected / total) * 100 : 0
  const done = inspected >= total && total > 0
  const circumference = 2 * Math.PI * 9

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "0 4px",
      }}
    >
      <div
        style={{
          position: "relative",
          width: 22,
          height: 22,
          flexShrink: 0,
        }}
      >
        <svg
          width={22}
          height={22}
          viewBox="0 0 22 22"
          style={{
            border: "none",
            outline: "none",
            margin: 0,
            padding: 0,
            minHeight: "unset",
            maxHeight: "unset",
            aspectRatio: "unset",
            display: "block",
          }}
        >
          <circle
            cx={11}
            cy={11}
            r={9}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={2.5}
          />
          <circle
            cx={11}
            cy={11}
            r={9}
            fill="none"
            stroke={done ? "#66BB6A" : "#7EB8FF"}
            strokeWidth={2.5}
            strokeDasharray={`${(pct / 100) * circumference} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(-90 11 11)"
            style={{ transition: "stroke-dasharray 0.3s ease" }}
          />
        </svg>
        {done && (
          <span
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 9,
              color: "#66BB6A",
            }}
          >
            ✓
          </span>
        )}
      </div>
      <span
        style={{
          fontSize: 10,
          fontFamily: FONT,
          color: done ? "#66BB6A" : "rgba(255,255,255,0.5)",
          fontWeight: 600,
          letterSpacing: "-0.02em",
        }}
      >
        {inspected}
        <span style={{ opacity: 0.4 }}>/{total}</span>
      </span>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main toolbar                                                       */
/* ------------------------------------------------------------------ */

interface ToolbarProps {
  showOnlyPolylines: boolean
  onTogglePolylines: () => void
  hideBbox: boolean
  onSetHideBbox: (v: boolean) => void
  animated: boolean
  onSetAnimated: (v: boolean) => void
  strokeScale: number[]
  onSetStrokeScale: (v: number[]) => void
  hideInspected: boolean
  onSetHideInspected: (v: boolean) => void
  inspectedCount: number
  totalCount: number
  onNavigateNext: () => void
  onRecenter: () => void
  onToggleSidebar: () => void
  onResetAll: () => void
  onSend: () => void
  selectedCount: number
  theme: any
}

export const Toolbar: React.FC<ToolbarProps> = ({
  showOnlyPolylines,
  onTogglePolylines,
  hideBbox,
  onSetHideBbox,
  animated,
  onSetAnimated,
  strokeScale,
  onSetStrokeScale,
  hideInspected,
  onSetHideInspected,
  inspectedCount,
  totalCount,
  onNavigateNext,
  onRecenter,
  onToggleSidebar,
  onResetAll,
  onSend,
  selectedCount,
  theme,
}) => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: 6,
        height: TOOLBAR_HEIGHT,
        padding: "6px 2px",
        overflowX: "auto",
        overflowY: "hidden",
        whiteSpace: "nowrap",
        // CSS vars for Switch component
        // @ts-ignore
        "--color-gray-100": theme?.colors?.backgroundTertiary ?? "#f0f0f0",
        "--color-gray-200": theme?.colors?.borderOpaque ?? "#e0e0e0",
        "--color-gray-500": theme?.colors?.borderSelected ?? "#b0b0b0",
        "--color-gray-700": theme?.colors?.contentPrimary ?? "#333333",
        "--color-blue": theme?.colors?.accent ?? "#276ef1",
      }}
    >
      {/* ── View ── */}
      <GroupPill label="View">
        <ToolBtn
          icon={Icons.layers}
          label={showOnlyPolylines ? "Image" : "Traces"}
          kbd="T"
          onClick={onTogglePolylines}
          title="Toggle between image and polyline-only view"
        />
        <ToolBtn
          icon={Icons.crosshair}
          label="Recenter"
          onClick={onRecenter}
          title="Reset zoom & pan"
        />
        <SwitchRow checked={hideBbox} onChange={onSetHideBbox} label="BBox" />
        <SwitchRow checked={animated} onChange={onSetAnimated} label="Anim" />
      </GroupPill>

      {/* ── Stroke ── */}
      <GroupPill label="Stroke">
        <div style={{ width: 140, padding: "0 6px" }}>
          <Slider
            value={strokeScale}
            max={5}
            min={0.1}
            step={0.1}
            persistentThumb
            onChange={({ value }) => value && onSetStrokeScale(value)}
            overrides={{
              Root: {
                style: { marginTop: 0, marginBottom: 0 },
              },
              TickBar: { style: () => ({ display: "none" }) },
              InnerThumb: { style: () => ({ display: "none" }) },
              ThumbValue: { style: () => ({ display: "none" }) },
              Thumb: {
                style: () => ({
                  width: "12px",
                  height: "12px",
                  borderRadius: "50%",
                  backgroundColor: "#7EB8FF",
                  borderColor: "transparent",
                  boxShadow: "0 0 6px rgba(126,184,255,0.35)",
                }),
              },
              Track: {
                style: () => ({
                  paddingTop: "4px",
                  paddingBottom: "4px",
                }),
              },
              InnerTrack: {
                style: () => ({
                  height: "3px",
                  borderRadius: "2px",
                  background: "rgba(126,184,255,0.3)",
                }),
              },
            }}
          />
        </div>
        <span
          style={{
            fontSize: 10,
            color: "rgba(255,255,255,0.35)",
            fontFamily: FONT,
            minWidth: 28,
            textAlign: "right",
            paddingRight: 4,
          }}
        >
          {strokeScale[0].toFixed(1)}×
        </span>
      </GroupPill>

      {/* ── Inspect ── */}
      <GroupPill label="Inspect">
        <ToolBtn
          icon={Icons.skipForward}
          label="Next"
          kbd="N"
          onClick={onNavigateNext}
          disabled={inspectedCount >= totalCount}
          accent
          title="Navigate to next uninspected fiber"
        />
        <SwitchRow
          checked={hideInspected}
          onChange={onSetHideInspected}
          label="Hide done"
        />
        <ProgressRing inspected={inspectedCount} total={totalCount} />
      </GroupPill>

      {/* ── Actions ── */}
      <GroupPill label="Actions">
        <ToolBtn
          icon={Icons.filter}
          label="Filter"
          kbd="F"
          onClick={onToggleSidebar}
          title="Toggle fiber type filter panel"
        />
        <ToolBtn
          icon={Icons.rotateCcw}
          label="Reset"
          onClick={onResetAll}
          title="Clear all selections and inspections"
        />
        <ToolBtn
          icon={Icons.send}
          label={`Send${selectedCount > 0 ? ` · ${selectedCount}` : ""}`}
          onClick={onSend}
          disabled={selectedCount === 0}
          accent
          title="Send selected fiber IDs to Streamlit"
        />
      </GroupPill>
    </div>
  )
}
