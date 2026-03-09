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
/*  Theme-aware color palette                                          */
/* ------------------------------------------------------------------ */

interface ThemeColors {
  fg: string
  fgHover: string
  fgActive: string
  fgMuted: string
  fgVeryMuted: string
  fgSubtle: string
  bgBtn: string
  bgBtnHover: string
  bgGroup: string
  borderGroup: string
  accent: string
  accentBg: string
  accentBgHover: string
  accentHover: string
  ring: string
  success: string
}

function buildColors(theme: any): ThemeColors {
  const bg = theme?.colors?.backgroundPrimary ?? "#000000"
  // const isDark = bg.charAt(1) < "8"
  const isDark = true
  if (isDark) {
    return {
      fg: "rgba(255,255,255,0.72)",
      fgHover: "#fff",
      fgActive: "rgba(255,255,255,0.85)",
      fgMuted: "rgba(255,255,255,0.4)",
      fgVeryMuted: "rgba(255,255,255,0.22)",
      fgSubtle: "rgba(255,255,255,0.35)",
      bgBtn: "rgba(255,255,255,0.05)",
      bgBtnHover: "rgba(255,255,255,0.12)",
      bgGroup: "rgba(255,255,255,0.03)",
      borderGroup: "rgba(255,255,255,0.06)",
      accent: "#7EB8FF",
      accentBg: "rgba(99,162,255,0.12)",
      accentBgHover: "rgba(99,162,255,0.25)",
      accentHover: "#9ECBFF",
      ring: "rgba(255,255,255,0.06)",
      success: "#66BB6A",
    }
  }

  return {
    fg: "rgba(0,0,0,0.65)",
    fgHover: "#000",
    fgActive: "rgba(0,0,0,0.85)",
    fgMuted: "rgba(0,0,0,0.4)",
    fgVeryMuted: "rgba(0,0,0,0.22)",
    fgSubtle: "rgba(0,0,0,0.35)",
    bgBtn: "rgba(0,0,0,0.04)",
    bgBtnHover: "rgba(0,0,0,0.08)",
    bgGroup: "rgba(0,0,0,0.02)",
    borderGroup: "rgba(0,0,0,0.08)",
    accent: "#1a6ddb",
    accentBg: "rgba(26,109,219,0.08)",
    accentBgHover: "rgba(26,109,219,0.15)",
    accentHover: "#1459b5",
    ring: "rgba(0,0,0,0.08)",
    success: "#2e7d32",
  }
}

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
  colors: ThemeColors
}

const ToolBtn: React.FC<ToolBtnProps> = ({
  icon,
  label,
  kbd,
  onClick,
  disabled,
  accent,
  title,
  colors,
}) => {
  const [hovered, setHovered] = React.useState(false)

  const bg = accent ? colors.accentBg : colors.bgBtn
  const bgHover = accent ? colors.accentBgHover : colors.bgBtnHover
  const fg = accent ? colors.accent : colors.fg
  const fgHover = accent ? colors.accentHover : colors.fgHover

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
  colors: ThemeColors
}> = ({ checked, onChange, label, colors }) => (
  <label
    style={{
      display: "flex",
      alignItems: "center",
      gap: 6,
      cursor: "pointer",
      fontSize: 11,
      color: checked ? colors.fgActive : colors.fgMuted,
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
  colors: ThemeColors
}> = ({ children, label, colors }) => (
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
          color: colors.fgVeryMuted,
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
        background: colors.bgGroup,
        border: `1px solid ${colors.borderGroup}`,
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
  colors: ThemeColors
}> = ({ inspected, total, colors }) => {
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
            stroke={colors.ring}
            strokeWidth={2.5}
          />
          <circle
            cx={11}
            cy={11}
            r={9}
            fill="none"
            stroke={done ? colors.success : colors.accent}
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
              color: colors.success,
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
          color: done ? colors.success : colors.fgSubtle,
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
  const colors = React.useMemo(() => buildColors(theme), [theme])

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
        backgroundColor: "rgba(30, 30, 30, 0.95)", // Force dark background
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
      <GroupPill label="View" colors={colors}>
        <ToolBtn
          icon={Icons.layers}
          label={showOnlyPolylines ? "Image" : "Traces"}
          kbd="T"
          onClick={onTogglePolylines}
          title="Toggle between image and polyline-only view"
          colors={colors}
        />
        <ToolBtn
          icon={Icons.crosshair}
          label="Recenter"
          onClick={onRecenter}
          title="Reset zoom & pan"
          colors={colors}
        />
        <SwitchRow
          checked={hideBbox}
          onChange={onSetHideBbox}
          label="BBox"
          colors={colors}
        />
        <SwitchRow
          checked={animated}
          onChange={onSetAnimated}
          label="Anim"
          colors={colors}
        />
      </GroupPill>

      {/* ── Stroke ── */}
      <GroupPill label="Stroke" colors={colors}>
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
                  backgroundColor: colors.accent,
                  borderColor: "transparent",
                  boxShadow: `0 0 6px ${colors.accentBg}`,
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
                  background: colors.accentBg,
                }),
              },
            }}
          />
        </div>
        <span
          style={{
            fontSize: 10,
            color: colors.fgSubtle,
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
      <GroupPill label="Inspect" colors={colors}>
        <ToolBtn
          icon={Icons.skipForward}
          label="Next"
          kbd="N"
          onClick={onNavigateNext}
          disabled={inspectedCount >= totalCount}
          accent
          title="Navigate to next uninspected fiber"
          colors={colors}
        />
        <SwitchRow
          checked={hideInspected}
          onChange={onSetHideInspected}
          label="Hide done"
          colors={colors}
        />
        <ProgressRing
          inspected={inspectedCount}
          total={totalCount}
          colors={colors}
        />
      </GroupPill>

      {/* ── Actions ── */}
      <GroupPill label="Actions" colors={colors}>
        <ToolBtn
          icon={Icons.filter}
          label="Filter"
          kbd="F"
          onClick={onToggleSidebar}
          title="Toggle fiber type filter panel"
          colors={colors}
        />
        <ToolBtn
          icon={Icons.rotateCcw}
          label="Reset"
          onClick={onResetAll}
          title="Clear all selections and inspections"
          colors={colors}
        />
        <ToolBtn
          icon={Icons.send}
          label={`Send${selectedCount > 0 ? ` · ${selectedCount}` : ""}`}
          onClick={onSend}
          disabled={selectedCount === 0}
          accent
          title="Send selected fiber IDs to Streamlit"
          colors={colors}
        />
      </GroupPill>
    </div>
  )
}
