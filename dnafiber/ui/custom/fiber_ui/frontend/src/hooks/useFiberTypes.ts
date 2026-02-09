import { useState, useCallback, useMemo } from "react"
import { Fiber } from "../types"

export function useFiberTypes(elements: Fiber[]) {
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set())

  const fiberTypes = useMemo(() => {
    const types = new Set<string>()
    elements.forEach((el) => {
      if (el.type) types.add(el.type)
    })
    return Array.from(types).sort()
  }, [elements])

  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    elements.forEach((el) => {
      const t = el.type || "(no type)"
      counts[t] = (counts[t] || 0) + 1
    })
    return counts
  }, [elements])

  const toggleType = useCallback((type: string) => {
    setHiddenTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }, [])

  const showAllTypes = useCallback(() => setHiddenTypes(new Set()), [])

  const hideAllTypes = useCallback(
    () => setHiddenTypes(new Set(fiberTypes)),
    [fiberTypes]
  )

  return {
    hiddenTypes,
    fiberTypes,
    typeCounts,
    toggleType,
    showAllTypes,
    hideAllTypes,
  }
}
