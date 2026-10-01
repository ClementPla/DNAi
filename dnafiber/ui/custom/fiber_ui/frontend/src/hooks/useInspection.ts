import { useState, useCallback, useRef, useEffect } from "react"
import { INSPECTION_DELAY } from "../utils"

const selectionKey = (ids: number[]) =>
  [...ids].sort((a, b) => a - b).join(",")

export function useInspection(committedSelection: number[] = []) {
  const [inspectedFibers, setInspectedFibers] = useState<Set<number>>(new Set())
  const [selectedFibers, setSelectedFibers] =
    useState<number[]>(committedSelection)

  // Resync with the selection committed on the Python side (e.g. after the
  // other viewer instance sent its selection, or after a remount).
  const committedKey = selectionKey(committedSelection)
  useEffect(() => {
    setSelectedFibers(committedKey ? committedKey.split(",").map(Number) : [])
  }, [committedKey])
  const isDirty = selectionKey(selectedFibers) !== committedKey
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
    isDirty,
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
