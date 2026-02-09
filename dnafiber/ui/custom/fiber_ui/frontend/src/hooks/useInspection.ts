import { useState, useCallback, useRef } from "react"
import { INSPECTION_DELAY } from "../utils"

export function useInspection() {
  const [inspectedFibers, setInspectedFibers] = useState<Set<number>>(new Set())
  const [selectedFibers, setSelectedFibers] = useState<number[]>([])
  const [hoveredFiberId, setHoveredFiberId] = useState<number | null>(null)
  const [hideInspected, setHideInspected] = useState(false)
  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null)

  const handleFiberMouseEnter = useCallback((fiberId: number) => {
    setHoveredFiberId(fiberId)
    hoverTimerRef.current = setTimeout(() => {
      setInspectedFibers((prev) => new Set(prev).add(fiberId))
    }, INSPECTION_DELAY)
  }, [])

  const handleFiberMouseLeave = useCallback(() => {
    setHoveredFiberId(null)
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current)
      hoverTimerRef.current = null
    }
  }, [])

  const handleFiberClick = useCallback((fiberId: number) => {
    setInspectedFibers((prev) => new Set(prev).add(fiberId))
    setSelectedFibers((prev) =>
      prev.includes(fiberId)
        ? prev.filter((i) => i !== fiberId)
        : [...prev, fiberId]
    )
  }, [])

  const markInspected = useCallback((fiberId: number) => {
    setInspectedFibers((prev) => new Set(prev).add(fiberId))
  }, [])

  const resetAll = useCallback(() => {
    setSelectedFibers([])
    setInspectedFibers(new Set())
  }, [])

  return {
    inspectedFibers,
    setInspectedFibers,
    selectedFibers,
    hoveredFiberId,
    hideInspected,
    setHideInspected,
    handleFiberMouseEnter,
    handleFiberMouseLeave,
    handleFiberClick,
    markInspected,
    resetAll,
  }
}
