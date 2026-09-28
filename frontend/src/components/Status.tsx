export function Loading({ what }: { what: string }) {
  return <div className="animate-pulse py-8 text-sm text-zinc-500">loading {what}…</div>;
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="my-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 font-mono text-sm text-red-400">
      {error}
    </div>
  );
}

export function UntrainedBanner() {
  return (
    <div className="mb-6 rounded-md border border-amber-800 bg-amber-950/30 px-4 py-2.5 text-sm text-amber-400">
      Untrained model — output is random. This visualizes the inference machinery, not language ability.
    </div>
  );
}

export function TrainedBanner({ corpus }: { corpus: string }) {
  return (
    <div className="mb-6 rounded-md border border-emerald-800 bg-emerald-950/30 px-4 py-2.5 text-sm text-emerald-400">
      Trained checkpoint loaded — this small model memorized{" "}
      <span className="font-mono">{corpus}</span>, so it echoes fragments of that text rather than
      producing novel language.
    </div>
  );
}
