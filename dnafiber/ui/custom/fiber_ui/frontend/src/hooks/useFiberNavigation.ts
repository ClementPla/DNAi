import { useCallback, RefObject } from "react"
import { ReactZoomPanPinchRef } from "react-zoom-pan-pinch"
import { Fiber } from "../types"
import { VIEWER_HEIGHT } from "../utils"

interface NavigationParams {
  elements: Fiber[]
  inspectedFibers: Set<number>
  setInspectedFibers: (s: Set<number>) => void
  hiddenTypes: Set<string>
  transformRef: RefObject<ReactZoomPanPinchRef | null>
  effectiveWidth: number
  svgWidth: number
  svgHeight: number
  image_w: number
  image_h: number
}

export function useFiberNavigation({
  elements,
  inspectedFibers,
  setInspectedFibers,
  hiddenTypes,
  transformRef,
  effectiveWidth,
  svgWidth,
  svgHeight,
  image_w,
  image_h,
}: NavigationParams) {
  return useCallback(() => {
    const uninspected = elements.find(
      (el) => !inspectedFibers.has(el.fiber_id) && !hiddenTypes.has(el.type)
    )
    if (!uninspected || !transformRef.current) return

    const next = new Set(inspectedFibers)
    next.add(uninspected.fiber_id)
    setInspectedFibers(next)

    const fiberVBX = uninspected.x + uninspected.width / 2
    const fiberVBY = uninspected.y + uninspected.height / 2

    const scaleX = svgWidth / image_w
    const scaleY = svgHeight / image_h
    const actualScale = Math.min(scaleX, scaleY)

    const renderedWidth = image_w * actualScale
    const renderedHeight = image_h * actualScale

    const svgInternalOffsetX = (svgWidth - renderedWidth) / 2
    const svgInternalOffsetY = (svgHeight - renderedHeight) / 2

    const fiberInSvgX = svgInternalOffsetX + fiberVBX * actualScale
    const fiberInSvgY = svgInternalOffsetY + fiberVBY * actualScale

    const contentOffsetX = (effectiveWidth - svgWidth) / 2
    const contentOffsetY = (VIEWER_HEIGHT - svgHeight) / 2

    const fiberContentX = contentOffsetX + fiberInSvgX
    const fiberContentY = contentOffsetY + fiberInSvgY

    const targetScale = 10
    const vpCenterX = effectiveWidth / 2
    const vpCenterY = VIEWER_HEIGHT / 2

    const newX = vpCenterX - fiberContentX * targetScale
    const newY = vpCenterY - fiberContentY * targetScale

    transformRef.current.setTransform(newX, newY, targetScale, 300)
  }, [
    elements,
    inspectedFibers,
    setInspectedFibers,
    hiddenTypes,
    transformRef,
    effectiveWidth,
    svgWidth,
    svgHeight,
    image_w,
    image_h,
  ])
}
