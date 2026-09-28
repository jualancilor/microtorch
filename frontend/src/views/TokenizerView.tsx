import { useEffect, useState } from "react";
import { api, displayToken, type TokenizeResponse } from "../api";
import { ErrorBox } from "../components/Status";

// deterministic per-id chip tint from a small fixed palette (id % 6)
const CHIP_COLORS = [
  "bg-emerald-950/60 border-emerald-900",
  "bg-zinc-800/80 border-zinc-700",
  "bg-emerald-900/30 border-emerald-950",
  "bg-zinc-900 border-zinc-700",
  "bg-emerald-950/30 border-zinc-700",
  "bg-zinc-800/40 border-zinc-800",
];

export default function TokenizerView() {
  const [text, setText] = useState<string>("To be, or not to be, that is the question.");
  const [result, setResult] = useState<TokenizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (text.length === 0) {
      setResult(null);
      setError(null);
      return;
    }
    const handle = window.setTimeout(() => {
      api
        .tokenize(text)
        .then((r) => {
          setResult(r);
          setError(null);
        })
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    }, 250);
    return () => window.clearTimeout(handle);
  }, [text]);

  return (
    <div className="max-w-3xl">
      <header className="mb-6">
        <h1 className="text-xl font-semibold text-zinc-100">Tokenizer</h1>
        <p className="mt-1 text-sm text-zinc-500">
          char-level: one token per character, vocabulary built from{" "}
          <span className="font-mono">data/sample.txt</span>
          {result && <span className="font-mono"> · vocab_size={result.vocab_size}</span>}
        </p>
      </header>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        spellCheck={false}
        className="w-full rounded-md border border-zinc-800 bg-zinc-900/60 p-3 font-mono text-sm text-zinc-200 outline-none focus:border-emerald-700"
        placeholder="type text to tokenize…"
      />

      {error && <ErrorBox error={error} />}

      {result && !error && (
        <>
          <div className="mb-2 mt-6 text-xs text-zinc-500">
            {result.ids.length} tokens — hover a chip for its id
          </div>
          <div className="flex flex-wrap gap-1.5">
            {result.tokens.map((tok, i) => (
              <span
                key={`${i}-${result.ids[i]}`}
                title={`id ${result.ids[i]}`}
                className={`inline-flex flex-col items-center rounded border px-1.5 py-0.5 font-mono ${CHIP_COLORS[result.ids[i] % CHIP_COLORS.length]}`}
              >
                <span className="text-sm text-zinc-100">{displayToken(tok)}</span>
                <span className="text-[10px] leading-tight text-zinc-500">{result.ids[i]}</span>
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
