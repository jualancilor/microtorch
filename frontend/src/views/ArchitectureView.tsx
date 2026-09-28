import { useEffect, useState } from "react";
import { api, type Architecture, type ArchNode, type ParamsResponse } from "../api";
import { ErrorBox, Loading } from "../components/Status";

const KIND_STYLE: Record<ArchNode["kind"], string> = {
  io: "border-zinc-700 border-dashed text-zinc-500",
  embedding: "border-emerald-900 bg-emerald-950/20",
  block: "border-zinc-700 bg-zinc-900/60",
  norm: "border-zinc-700 bg-zinc-900/30",
  head: "border-emerald-900 bg-emerald-950/20",
};

function fmt(n: number): string {
  return n.toLocaleString("en-US");
}

function NodeCard({ node, expanded }: { node: ArchNode; expanded: boolean }) {
  return (
    <div className={`w-full rounded-md border px-4 py-3 ${KIND_STYLE[node.kind]}`}>
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-sm font-medium text-zinc-200">{node.label}</span>
        <span className="font-mono text-xs text-zinc-500">
          {node.params > 0 ? `${fmt(node.params)} params` : "—"}
        </span>
      </div>
      <div className="mt-0.5 font-mono text-xs text-zinc-500">{node.detail}</div>
      {expanded && node.children && (
        <div className="mt-3 space-y-1.5 border-t border-zinc-800 pt-3">
          {node.children.map((c) => (
            <div key={c.label} className="flex items-baseline justify-between gap-4 pl-3">
              <span className="text-xs text-zinc-400">{c.label}</span>
              <span className="shrink-0 font-mono text-xs text-zinc-600">
                {c.detail} · {fmt(c.params)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ArchitectureView() {
  const [arch, setArch] = useState<Architecture | null>(null);
  const [params, setParams] = useState<ParamsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.architecture(), api.params()])
      .then(([a, p]) => {
        setArch(a);
        setParams(p);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <ErrorBox error={error} />;
  if (!arch || !params) return <Loading what="architecture" />;

  const cfg = arch.config;

  return (
    <div className="max-w-5xl">
      <header className="mb-8">
        <h1 className="text-xl font-semibold text-zinc-100">Architecture</h1>
        <p className="mt-1 text-sm text-zinc-500">
          decoder-only GPT ·{" "}
          <span className="font-mono">
            L={cfg.n_layer} · C={cfg.n_embd} · H={cfg.n_head} · T={cfg.block_size} · V={cfg.vocab_size}
          </span>
        </p>
        <div className="mt-4 font-mono text-3xl font-semibold text-emerald-400">
          {fmt(arch.total_params)} <span className="text-base text-zinc-500">parameters</span>
        </div>
      </header>

      <div className="flex gap-10">
        <div className="flex max-w-lg flex-1 flex-col items-center">
          {arch.nodes.map((node, i) => (
            <div key={node.id} className="flex w-full flex-col items-center">
              {i > 0 && <div className="h-5 w-px bg-zinc-700" />}
              {/* all blocks are identical; expand only the first to keep the diagram readable */}
              <NodeCard node={node} expanded={node.id === "block0"} />
            </div>
          ))}
        </div>

        <aside className="w-72 shrink-0">
          <h2 className="mb-3 text-sm font-medium text-zinc-400">Per-module breakdown</h2>
          <table className="w-full text-xs">
            <tbody>
              {params.layers.map((l) => (
                <tr key={l.name} className="border-b border-zinc-900">
                  <td className="py-1.5 pr-2 font-mono text-zinc-400">{l.name}</td>
                  <td className="py-1.5 text-right font-mono text-zinc-500">{fmt(l.params)}</td>
                </tr>
              ))}
              <tr>
                <td className="py-2 pr-2 font-mono font-semibold text-zinc-200">total</td>
                <td className="py-2 text-right font-mono font-semibold text-emerald-400">
                  {fmt(params.total)}
                </td>
              </tr>
            </tbody>
          </table>
        </aside>
      </div>
    </div>
  );
}
