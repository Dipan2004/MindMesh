import { cn } from '@/lib/utils'

/**
 * MindMesh brand mark: three linked nodes (a tiny knowledge graph) on a solid
 * tile. Monochrome: the tile takes the current text colour and the nodes take
 * the page background, so it reads black-on-light and white-on-dark.
 */
export default function BrandMark({
  className,
  nodeColor = 'var(--background)',
}: {
  className?: string
  /** Colour of the nodes/links drawn on the tile (defaults to the page background). */
  nodeColor?: string
}) {
  return (
    <svg viewBox="0 0 32 32" className={cn('size-6 shrink-0 text-foreground', className)} aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="currentColor" />
      <g style={{ stroke: nodeColor }} strokeWidth="1.8" strokeLinecap="round" strokeOpacity="0.9">
        <line x1="10" y1="11" x2="22" y2="10" />
        <line x1="10" y1="11" x2="15" y2="22" />
        <line x1="22" y1="10" x2="15" y2="22" />
      </g>
      <g style={{ fill: nodeColor }}>
        <circle cx="10" cy="11" r="3" />
        <circle cx="22" cy="10" r="2.6" />
        <circle cx="15" cy="22" r="3.4" />
      </g>
    </svg>
  )
}
