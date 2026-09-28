import { displayToken, type GenCandidate } from "../api";

interface Props {
  candidates: GenCandidate[];
  chosenId: number;
}

/** Horizontal SVG bar chart of top-k candidate probabilities for one step. */
export default function ProbBarChart({ candidates, chosenId }: Props) {
  const rowH = 24;
  const labelW = 52;
  const chartW = 320;
  const valueW = 64;
  const width = labelW + chartW + valueW;
  const height = candidates.length * rowH;
  const maxP = Math.max(...candidates.map((c) => c.prob), 1e-9);

  return (
    <svg width={width} height={height} role="img" aria-label="top-k probabilities">
      {candidates.map((c, i) => {
        const y = i * rowH;
        const w = Math.max(1, (c.prob / maxP) * chartW);
        const chosen = c.token_id === chosenId;
        return (
          <g key={c.token_id} transform={`translate(0, ${y})`}>
            <text
              x={labelW - 8}
              y={rowH / 2 + 4}
              textAnchor="end"
              className="fill-zinc-300"
              fontFamily="monospace"
              fontSize={12}
            >
              {displayToken(c.token)}
            </text>
            <rect
              x={labelW}
              y={4}
              width={w}
              height={rowH - 8}
              rx={2}
              className={chosen ? "fill-emerald-500" : "fill-zinc-700"}
            />
            <text
              x={labelW + w + 8}
              y={rowH / 2 + 4}
              className={chosen ? "fill-emerald-400" : "fill-zinc-500"}
              fontFamily="monospace"
              fontSize={11}
            >
              {(c.prob * 100).toFixed(2)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
