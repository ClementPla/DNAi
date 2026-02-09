import React from "react"
import { Switch } from "@base-ui-components/react/switch"
import switch_styles from "../MySwitch.module.css"
import { SIDEBAR_WIDTH, VIEWER_HEIGHT } from "../utils"

interface SidebarProps {
  open: boolean
  fiberTypes: string[]
  typeCounts: Record<string, number>
  hiddenTypes: Set<string>
  onToggleType: (type: string) => void
  onShowAll: () => void
  onHideAll: () => void
  visibleCount: number
  totalCount: number
}

export const Sidebar: React.FC<SidebarProps> = ({
  open,
  fiberTypes,
  typeCounts,
  hiddenTypes,
  onToggleType,
  onShowAll,
  onHideAll,
  visibleCount,
  totalCount,
}) => {
  return (
    <div
      style={{
        width: open ? SIDEBAR_WIDTH : 0,
        height: VIEWER_HEIGHT,
        overflow: "hidden",
        transition: "width 0.25s ease",
        flexShrink: 0,
      }}
    >
      <div
        style={{
          width: SIDEBAR_WIDTH,
          height: "100%",
          backgroundColor: "rgba(30, 30, 30, 0.95)",
          borderLeft: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "0 8px 8px 0",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "12px 12px 8px",
            borderBottom: "1px solid rgba(255,255,255,0.1)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span
            style={{
              color: "white",
              fontSize: 13,
              fontWeight: 600,
              fontFamily: "system-ui",
            }}
          >
            Fiber Types
          </span>
          <div style={{ display: "flex", gap: 4 }}>
            <button
              onClick={onShowAll}
              style={{
                background: "none",
                border: "1px solid rgba(255,255,255,0.2)",
                color: "rgba(255,255,255,0.7)",
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 3,
                cursor: "pointer",
              }}
            >
              All
            </button>
            <button
              onClick={onHideAll}
              style={{
                background: "none",
                border: "1px solid rgba(255,255,255,0.2)",
                color: "rgba(255,255,255,0.7)",
                fontSize: 10,
                padding: "2px 6px",
                borderRadius: 3,
                cursor: "pointer",
              }}
            >
              None
            </button>
          </div>
        </div>

        {/* Type list */}
        <div style={{ flex: 1, overflowY: "auto", padding: "8px 12px" }}>
          {fiberTypes.map((type) => {
            const isVisible = !hiddenTypes.has(type)
            const count = typeCounts[type] || 0
            return (
              <div
                key={type}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "6px 0",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                  cursor: "pointer",
                  opacity: isVisible ? 1 : 0.4,
                  transition: "opacity 0.15s ease",
                }}
                onClick={() => onToggleType(type)}
              >
                <Switch.Root
                  checked={isVisible}
                  onCheckedChange={() => onToggleType(type)}
                  className={switch_styles.Switch}
                >
                  <Switch.Thumb className={switch_styles.Thumb} />
                </Switch.Root>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    minWidth: 0,
                    flex: 1,
                  }}
                >
                  <span
                    style={{
                      color: "white",
                      fontSize: 12,
                      fontFamily: "system-ui",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                    title={type}
                  >
                    {type}
                  </span>
                  <span
                    style={{
                      color: "rgba(255,255,255,0.4)",
                      fontSize: 10,
                      fontFamily: "system-ui",
                    }}
                  >
                    {count} fiber{count !== 1 ? "s" : ""}
                  </span>
                </div>
              </div>
            )
          })}

          {fiberTypes.length === 0 && (
            <div
              style={{
                color: "rgba(255,255,255,0.4)",
                fontSize: 12,
                fontFamily: "system-ui",
                textAlign: "center",
                paddingTop: 20,
              }}
            >
              No fiber types found
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: "8px 12px",
            borderTop: "1px solid rgba(255,255,255,0.1)",
            fontSize: 10,
            color: "rgba(255,255,255,0.5)",
            fontFamily: "system-ui",
          }}
        >
          Showing {visibleCount}/{totalCount} fibers
          {hiddenTypes.size > 0 && (
            <span> · {hiddenTypes.size} type(s) hidden</span>
          )}
        </div>
      </div>
    </div>
  )
}
