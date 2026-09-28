import { useState } from "react";
import ArchitectureView from "./views/ArchitectureView";
import AttentionView from "./views/AttentionView";
import GenerateView from "./views/GenerateView";
import TokenizerView from "./views/TokenizerView";

type ViewId = "architecture" | "tokenizer" | "generate" | "attention";

const NAV: { id: ViewId; label: string; hint: string }[] = [
  { id: "architecture", label: "Architecture", hint: "layer graph + params" },
  { id: "tokenizer", label: "Tokenizer", hint: "char → id" },
  { id: "generate", label: "Generate", hint: "sampling, step by step" },
  { id: "attention", label: "Attention", hint: "head heatmaps" },
];

export default function App() {
  const [view, setView] = useState<ViewId>("architecture");

  return (
    <div className="flex min-h-screen">
      <nav className="w-56 shrink-0 border-r border-zinc-800 p-4 flex flex-col gap-1">
        <div className="px-3 pb-4 pt-2">
          <div className="font-mono text-lg font-semibold text-zinc-100">microgpt</div>
          <div className="text-xs text-zinc-500">on microtorch · untrained</div>
        </div>
        {NAV.map((item) => (
          <button
            key={item.id}
            onClick={() => setView(item.id)}
            className={`rounded-md px-3 py-2 text-left transition-colors ${
              view === item.id
                ? "bg-zinc-900 text-emerald-400"
                : "text-zinc-400 hover:bg-zinc-900/60 hover:text-zinc-200"
            }`}
          >
            <div className="text-sm font-medium">{item.label}</div>
            <div className="text-xs text-zinc-600">{item.hint}</div>
          </button>
        ))}
        <div className="mt-auto px-3 py-2 text-xs leading-relaxed text-zinc-600">
          5–8M param decoder-only GPT with random weights. Backend:{" "}
          <span className="font-mono">uvicorn backend.main:app</span>
        </div>
      </nav>
      <main className="min-w-0 flex-1 p-8">
        {view === "architecture" && <ArchitectureView />}
        {view === "tokenizer" && <TokenizerView />}
        {view === "generate" && <GenerateView />}
        {view === "attention" && <AttentionView />}
      </main>
    </div>
  );
}
