import { Fiber } from "../types"
import {
  parsePolylinePoints,
  calculatePolylineLength,
  formatLength,
  formatLengthPrecise,
  getScaleInfo,
  isGreen,
  isRed,
} from "../utils"
import { ExportOptions } from "../components/ContextMenu"

interface ExportFiberParams {
  fiber: Fiber
  imageSrc: string
  imageW: number
  imageH: number
  pixelSize: number
  currentScale: number
  fitScale: number
  margin: number
  strokeScale: number
  options: ExportOptions
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

interface Segment {
  pts: { x: number; y: number }[]
  lengthPx: number
  lengthUm: number
  color: string
}

function computeSegments(fiber: Fiber, pixelSize: number): Segment[] {
  const segments = fiber.points.map((line, i) => {
    const pts = parsePolylinePoints(line)
    const lengthPx = calculatePolylineLength(pts)
    const lengthUm = lengthPx * pixelSize
    return { pts, lengthPx, lengthUm, color: fiber.colors[i] }
  })

  const firstPt = segments[0]?.pts[0]
  const lastSeg = segments[segments.length - 1]
  const lastPt = lastSeg?.pts[lastSeg.pts.length - 1]
  const shouldReverse = firstPt && lastPt && firstPt.x > lastPt.x

  return shouldReverse ? [...segments].reverse() : segments
}

function drawPolyline(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number }[],
  offsetX: number,
  offsetY: number
) {
  if (pts.length < 2) return
  ctx.beginPath()
  ctx.moveTo(pts[0].x - offsetX, pts[0].y - offsetY)
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i].x - offsetX, pts[i].y - offsetY)
  }
  ctx.stroke()
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + r)
  ctx.lineTo(x + w, y + h - r)
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  ctx.lineTo(x + r, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}

export async function exportFiberAsPng(
  params: ExportFiberParams
): Promise<void> {
  const {
    fiber,
    imageSrc,
    imageW,
    imageH,
    pixelSize,
    currentScale,
    fitScale,
    margin,
    strokeScale,
    options,
  } = params

  const {
    includeScaleRuler,
    includeFlattenedFiber,
    segmentationOffset,
    bboxExpand,
  } = options

  const img = await loadImage(imageSrc)

  // --- Crop region around fiber ---
  const basePad = Math.max(margin * 5, 30)
  const pad = basePad * bboxExpand
  const cropX = Math.max(0, Math.floor(fiber.x - pad))
  const cropY = Math.max(0, Math.floor(fiber.y - pad))
  const cropRight = Math.min(imageW, Math.ceil(fiber.x + fiber.width + pad))
  const cropBottom = Math.min(imageH, Math.ceil(fiber.y + fiber.height + pad))
  const cropW = cropRight - cropX
  const cropH = cropBottom - cropY

  // --- Flattened fiber dimensions ---
  const segments = computeSegments(fiber, pixelSize)
  const totalLengthPx = segments.reduce((a, s) => a + s.lengthPx, 0)
  const totalLengthUm = segments.reduce((a, s) => a + s.lengthUm, 0)
  const greenLength = segments
    .filter((s) => isGreen(s.color))
    .reduce((a, s) => a + s.lengthUm, 0)
  const redLength = segments
    .filter((s) => isRed(s.color))
    .reduce((a, s) => a + s.lengthUm, 0)
  const ratio = redLength > 0 ? (greenLength / redLength).toFixed(2) : "∞"

  const barMaxWidth = Math.max(cropW * 0.8, totalLengthPx)
  const barScale = barMaxWidth / totalLengthPx
  const barHeight = 14
  const sectionPadding = 16
  const flattenedTotalHeight = includeFlattenedFiber
    ? sectionPadding + barHeight + 24 + sectionPadding
    : 0

  // --- Scale ruler ---
  const { barPx, label } = getScaleInfo(pixelSize, currentScale, fitScale)

  // --- Canvas setup ---
  const canvasW = includeFlattenedFiber
    ? Math.max(cropW, barMaxWidth + 40)
    : cropW
  const canvasH = cropH + flattenedTotalHeight
  const canvas = document.createElement("canvas")
  canvas.width = canvasW
  canvas.height = canvasH
  const ctx = canvas.getContext("2d")!

  // --- Background ---
  ctx.fillStyle = "#00000000"
  ctx.fillRect(0, 0, canvasW, canvasH)

  // --- Draw cropped image (centered if canvas wider) ---
  const imageOffsetX = Math.floor((canvasW - cropW) / 2)
  ctx.drawImage(img, cropX, cropY, cropW, cropH, imageOffsetX, 0, cropW, cropH)

  // --- Draw polylines with dashed stroke + offset ---
  const dashPattern = [8, 5]
  const baseWidth = (Math.min(imageW, imageH) / 1024) * strokeScale

  for (let i = 0; i < fiber.points.length; i++) {
    const pts = parsePolylinePoints(fiber.points[i])
    const color = fiber.colors[i]

    // Apply segmentation offset: shift polylines vertically
    const shiftedPts = pts.map((p) => ({
      x: p.x,
      y: p.y + segmentationOffset,
    }))

    // White outline (dashed)
    ctx.save()
    ctx.setLineDash(dashPattern)
    ctx.lineWidth = baseWidth * 2.2
    ctx.lineCap = "round"
    ctx.lineJoin = "round"
    ctx.strokeStyle = "white"
    drawPolyline(ctx, shiftedPts, cropX - imageOffsetX, cropY)
    ctx.restore()

    // Colored line (dashed)
    ctx.save()
    ctx.setLineDash(dashPattern)
    ctx.lineWidth = baseWidth * 1.4
    ctx.lineCap = "round"
    ctx.lineJoin = "round"
    ctx.strokeStyle = color
    drawPolyline(ctx, shiftedPts, cropX - imageOffsetX, cropY)
    ctx.restore()
  }

  // --- Scale ruler (overlaid on image, bottom-left corner) ---
  if (includeScaleRuler) {
    const rulerX = imageOffsetX + 12
    const rulerY = cropH - 40

    // Label
    ctx.fillStyle = "white"
    ctx.font = "12px system-ui, sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "top"
    ctx.fillText(label, rulerX, rulerY)

    // Bar
    const barY = rulerY + 18
    ctx.fillStyle = "white"
    ctx.setLineDash([])
    roundRect(ctx, rulerX, barY, barPx, 4, 2)
    ctx.fill()

    // End ticks
    ctx.fillRect(rulerX, barY - 4, 2, 12)
    ctx.fillRect(rulerX + barPx - 2, barY - 4, 2, 12)
  }

  // --- Flattened fiber bar (below image) ---
  if (includeFlattenedFiber) {
    const flatY = cropH + sectionPadding
    const barStartX = (canvasW - barMaxWidth) / 2
    let curX = barStartX

    // Background pill
    ctx.fillStyle = "rgba(0,0,0,0.7)"
    roundRect(
      ctx,
      barStartX - 6,
      flatY - 6,
      barMaxWidth + 12,
      barHeight + 30,
      6
    )
    ctx.fill()

    // Segments
    ctx.setLineDash([])
    for (const seg of segments) {
      const w = seg.lengthPx * barScale
      ctx.fillStyle = seg.color
      ctx.fillRect(curX, flatY, w, barHeight)

      ctx.strokeStyle = "rgba(0,0,0,0.3)"
      ctx.lineWidth = 1
      ctx.strokeRect(curX, flatY, w, barHeight)

      if (w > 30) {
        ctx.fillStyle = "white"
        ctx.font = "bold 10px system-ui, sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.shadowColor = "black"
        ctx.shadowBlur = 3
        ctx.fillText(
          formatLength(seg.lengthUm),
          curX + w / 2,
          flatY + barHeight / 2
        )
        ctx.shadowBlur = 0
      }

      curX += w
    }

    // Total + ratio label
    const labelY = flatY + barHeight + 6
    ctx.font = "11px system-ui, sans-serif"
    ctx.textBaseline = "top"

    const totalText = `Total: ${formatLengthPrecise(totalLengthUm)}`
    const ratioPrefix = "  ·  "
    const ratioSuffix = `: ${ratio}`

    const totalW = ctx.measureText(totalText).width
    const prefixW = ctx.measureText(ratioPrefix).width
    const gW = ctx.measureText("G").width
    const slashW = ctx.measureText("/").width
    const rW = ctx.measureText("R").width
    const suffixW = ctx.measureText(ratioSuffix).width
    const fullWidth = totalW + prefixW + gW + slashW + rW + suffixW
    const textStartX = canvasW / 2 - fullWidth / 2

    ctx.textAlign = "left"
    let tx = textStartX

    ctx.fillStyle = "white"
    ctx.fillText(totalText, tx, labelY)
    tx += totalW

    ctx.fillStyle = "white"
    ctx.fillText(ratioPrefix, tx, labelY)
    tx += prefixW

    ctx.fillStyle = "#90EE90"
    ctx.fillText("G", tx, labelY)
    tx += gW

    ctx.fillStyle = "white"
    ctx.fillText("/", tx, labelY)
    tx += slashW

    ctx.fillStyle = "#FF6B6B"
    ctx.fillText("R", tx, labelY)
    tx += rW

    ctx.fillStyle = "white"
    ctx.fillText(ratioSuffix, tx, labelY)
  }

  // --- Download ---
  canvas.toBlob((blob) => {
    if (!blob) return
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.download = `fiber_${fiber.fiber_id}.png`
    a.href = url
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, "image/png")
}
