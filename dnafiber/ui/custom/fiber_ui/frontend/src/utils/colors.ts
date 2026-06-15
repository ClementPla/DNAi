export const normalizeColor = (c: string): string => {
  const s = c.trim().toLowerCase()
  if (/^#[0-9a-f]{3}$/.test(s)) {
    return `#${s[1]}${s[1]}${s[2]}${s[2]}${s[3]}${s[3]}`
  }
  return s
}
