import { useEffect, useState } from "react";
import { api, displayToken, type GenerateResponse, type StatusResponse } from "../api";
import ProbBarChart from "../components/ProbBarChart";
import { ErrorBox, TrainedBanner, UntrainedBanner } from "../components/Status";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}

function Slider({ label, value, min, max, step, format, onChange }: SliderProps) {
  return (
    <label className="block">
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono text-emerald-400">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-emerald-500"
      />
    </label>
  );
}

export default function GenerateView() {
  const [prompt, setPrompt] = useState<string>("Shall I compare thee");
  const [temperature, setTemperature] = useState<number>(1.0);
  const [topK, setTopK] = useState<number>(10);
  const [maxNewTokens, setMaxNewTokens] = useState<number>(60);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [selectedStep, setSelectedStep] = useState<number>(0);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    api
      .status()
      .then(setStatus)
      .catch(() => setStatus(null)); // banner just falls back to the untrained warning
  }, []);

  const run = () => {
    setBusy(true);
    setError(null);
    api
      .generate({ prompt, max_new_tokens: maxNewTokens, temperature, top_k: topK })
      .then((r) => {
        setResult(r);
        setSelectedStep(0);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false));
  };

  const step = result?.steps[selectedStep];

  return (
    <div className="max-w-4xl">
      <header className="mb-6">
        <h1 className="text-xl font-semibold text-zinc-100">Generate</h1>
        <p className="mt-1 text-sm text-zinc-500">
          autoregressive sampling — click a generated token to inspect its top-k distribution
        </p>
      </header>

      {status?.trained ? <TrainedBanner corpus={status.corpus} /> : <UntrainedBanner />}

      <div className="grid grid-cols-[1fr_240px] gap-6">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          spellCheck={false}
          className="w-full rounded-md border border-zinc-800 bg-zinc-900/60 p-3 font-mono text-sm text-zinc-200 outline-none focus:border-emerald-700"
          placeholder="prompt…"
        />
        <div className="space-y-4">
          <Slider
            label="temperature"
            value={temperature}
            min={0}
            max={2}
            step={0.05}
            format={(v) => v.toFixed(2)}
            onChange={setTemperature}
          />
          <Slider
            label="top_k"
            value={topK}
            min={1}
            max={50}
            step={1}
            format={(v) => String(v)}
            onChange={setTopK}
          />
          <Slider
            label="max_new_tokens"
            value={maxNewTokens}
            min={1}
            max={200}
            step={1}
            format={(v) => String(v)}
            onChange={setMaxNewTokens}
          />
          <button
            onClick={run}
            disabled={busy || prompt.length === 0}
            className="w-full rounded-md bg-emerald-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-600 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
          >
            {busy ? "sampling…" : "generate"}
          </button>
        </div>
      </div>

      {error && <ErrorBox error={error} />}

      {result && !error && (
        <div className="mt-8 space-y-6">
          <section>
            <h2 className="mb-2 text-sm font-medium text-zinc-400">Output</h2>
            <div className="whitespace-pre-wrap rounded-md border border-zinc-800 bg-zinc-900/40 p-4 font-mono text-sm leading-relaxed">
              <span className="text-zinc-500">{result.prompt}</span>
              <span className="text-emerald-300">{result.output}</span>
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-sm font-medium text-zinc-400">
              Sampling steps — step {selectedStep + 1}/{result.steps.length}
            </h2>
            <div className="mb-4 flex flex-wrap gap-1">
              {result.steps.map((s, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedStep(i)}
                  title={`p=${(s.prob * 100).toFixed(2)}%`}
                  className={`rounded border px-1.5 py-0.5 font-mono text-sm transition-colors ${
                    i === selectedStep
                      ? "border-emerald-600 bg-emerald-950/60 text-emerald-300"
                      : "border-zinc-800 bg-zinc-900/50 text-zinc-400 hover:border-zinc-600"
                  }`}
                >
                  {displayToken(s.token)}
                </button>
              ))}
            </div>
            {step && (
              <div className="rounded-md border border-zinc-800 bg-zinc-900/40 p-4">
                <div className="mb-3 text-xs text-zinc-500">
                  candidates after temperature {temperature.toFixed(2)} + top-k filter; sampled token in
                  green
                </div>
                <ProbBarChart candidates={step.top_k} chosenId={step.token_id} />
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
