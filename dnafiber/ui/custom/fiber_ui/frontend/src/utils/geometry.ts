export function parsePolylinePoints(
  pointsStr: string
): { x: number; y: number }[] {
  return pointsStr
    .split(" ")
    .filter(Boolean)
    .map((p) => {
      const [x, y] = p.split(",").map(Number)
      return { x, y }
    })
}

export function calculatePolylineLength(
  points: { x: number; y: number }[]
): number {
  let length = 0
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x
    const dy = points[i].y - points[i - 1].y
    length += Math.sqrt(dx * dx + dy * dy)
  }
  return length
}

export function getScaleInfo(
  pixelSize: number,
  currentScale: number,
  baseScale: number
) {
  const actualScreenPixelsPerUnit = baseScale * currentScale
  const micronsPerScreenPixel = pixelSize / actualScreenPixelsPerUnit

  const barPx = 100
  const totalUm = barPx * micronsPerScreenPixel

  const label =
    totalUm >= 1000
      ? `${(totalUm / 1000).toPrecision(3)} mm`
      : `${totalUm.toPrecision(3)} μm`

  return { barPx, label }
}

export function formatLength(um: number): string {
  return um >= 1000 ? `${(um / 1000).toFixed(1)}mm` : `${um.toFixed(1)}μm`
}

export function formatLengthPrecise(um: number): string {
  return um >= 1000 ? `${(um / 1000).toFixed(2)} mm` : `${um.toFixed(1)} μm`
}

export const isGreen = (color: string) =>
  color.toLowerCase().includes("green") ||
  color === "#00ff00" ||
  color === "#00FF00"

export const isRed = (color: string) =>
  color.toLowerCase().includes("red") ||
  color === "#ff0000" ||
  color === "#f00" ||
  color === "#FF0000"
