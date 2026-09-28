import { useState } from "react";
import { displayToken } from "../api";

interface Props {
  tokens: string[];
  /** [query position][key position], rows sum to 1 */
  weights: number[][];
}

interface Hover {
  q: number;
  k: number;
  w: number;
}

/** T×T SVG heatmap; cell brightness = attention weight. */
export default function AttentionHeatmap({ tokens, weights }: Props) {
  const [hover, setHover] = useState<Hover | null>(null);
  const T = tokens.length;
  const cell = Math.max(8, Math.min(28, Math.floor(560 / T)));
  const labelPad = 26;
  const size = labelPad + T * cell;

  return (
    <div>
      <svg width={size} height={size} role="img" aria-label="attention heatmap">
        {tokens.map((tok, i) => (
          <g key={`lab-${i}`}>
            {/* column labels (keys) */}
            <text
              x={labelPad + i * cell + cell / 2}
              y={labelPad - 8}
              textAnchor="middle"
              fontFamily="monospace"
              fontSize={Math.min(11, cell - 2)}
              className="fill-zinc-500"
            >
              {displayToken(tok)}
            </text>
            {/* row labels (queries) */}
            <text
              x={labelPad - 8}
              y={labelPad + i * cell + cell / 2 + 4}
              textAnchor="end"
              fontFamily="monospace"
              fontSize={Math.min(11, cell - 2)}
              className="fill-zinc-500"
            >
              {displayToken(tok)}
            </text>
          </g>
        ))}
        {weights.map((row, q) =>
          row.map((w, k) => (
            <rect
              key={`${q}-${k}`}
              x={labelPad + k * cell}
              y={labelPad + q * cell}
              width={cell - 1}
              height={cell - 1}
              fill={
                k > q
                  ? "transparent" // masked future positions
                  : `rgba(52, 211, 153, ${Math.min(1, w)})` // emerald-400
              }
              stroke={hover && hover.q === q && hover.k === k ? "#fafafa" : "#27272a"}
              strokeWidth={hover && hover.q === q && hover.k === k ? 1.5 : 0.5}
              onMouseEnter={() => setHover({ q, k, w })}
              onMouseLeave={() => setHover(null)}
            />
          )),
        )}
      </svg>
      <div className="mt-2 h-5 font-mono text-xs text-zinc-400">
        {hover
          ? hover.k > hover.q
            ? `q "${displayToken(tokens[hover.q])}" → k "${displayToken(tokens[hover.k])}": masked (future)`
            : `q "${displayToken(tokens[hover.q])}" [${hover.q}] → k "${displayToken(tokens[hover.k])}" [${hover.k}] : ${hover.w.toFixed(4)}`
          : "hover a cell for token → token weight"}
      </div>
    </div>
  );
}
