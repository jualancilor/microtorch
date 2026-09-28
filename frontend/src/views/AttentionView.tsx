import { useState } from "react";
import { api, type AttentionResponse } from "../api";
import AttentionHeatmap from "../components/AttentionHeatmap";
import { ErrorBox } from "../components/Status";

const MAX_CHARS = 64;

function Picker({
  label,
  count,
  value,
  onChange,
}: {
  label: string;
  count: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 text-xs text-zinc-400">{label}</div>
      <div className="flex flex-wrap gap-1">
        {Array.from({ length: count }, (_, i) => (
          <button
            key={i}
            onClick={() => onChange(i)}
            className={`h-7 w-7 rounded border font-mono text-xs transition-colors ${
              i === value
                ? "border-emerald-600 bg-emerald-950/60 text-emerald-300"
                : "border-zinc-800 bg-zinc-900/50 text-zinc-500 hover:border-zinc-600"
            }`}
          >
            {i}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function AttentionView() {
  const [text, setText] = useState<string>("To be, or not to be");
  const [result, setResult] = useState<AttentionResponse | null>(null);
  const [layer, setLayer] = useState<number>(0);
  const [head, setHead] = useState<number>(0);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setBusy(true);
    setError(null);
    api
      .attention(text)
      .then((r) => {
        setResult(r);
        setLayer(0);
        setHead(0);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  };

  return (
    <div className="max-w-4xl">
      <header className="mb-6">
        <h1 className="text-xl font-semibold text-zinc-100">Attention</h1>
        <p className="mt-1 text-sm text-zinc-500">
          softmax weights from one forward pass — row = query token, column = key token; upper triangle
          is causally masked
        </p>
      </header>

      <div className="flex gap-3">
        <input
          value={text}
          onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
          maxLength={MAX_CHARS}
          spellCheck={false}
          className="flex-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2 font-mono text-sm text-zinc-200 outline-none focus:border-emerald-700"
          placeholder={`text (max ${MAX_CHARS} chars)…`}
        />
        <button
          onClick={run}
          disabled={busy || text.length === 0}
          className="rounded-md bg-emerald-700 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-600 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
        >
          {busy ? "running…" : "run forward pass"}
        </button>
      </div>
      <div className="mt-1 text-right font-mono text-xs text-zinc-600">
        {text.length}/{MAX_CHARS}
      </div>

      {error && <ErrorBox error={error} />}

      {result && !error && (
        <div className="mt-6 space-y-5">
          <div className="flex flex-wrap gap-8">
            <Picker label="layer" count={result.n_layer} value={layer} onChange={setLayer} />
            <Picker label="head" count={result.n_head} value={head} onChange={setHead} />
          </div>
          <div className="rounded-md border border-zinc-800 bg-zinc-900/40 p-4">
            <AttentionHeatmap tokens={result.tokens} weights={result.weights[layer][head]} />
          </div>
        </div>
      )}
    </div>
  );
}
