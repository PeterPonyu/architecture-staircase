import type { ReactNode } from "react";
import { Spine } from "./Spine";

const GITHUB = "https://github.com/PeterPonyu/architecture-staircase";
const CONCEPT = "10.5281/zenodo.21020348";
const VERSION = "10.5281/zenodo.21882597";

export function Notebook({ children }: { children: ReactNode }) {
  return (
    <div className="notebook" aria-label="Two-probe architecture staircase">
      <Spine />
      <header className="masthead">
        <p className="kicker">architecture-staircase</p>
        <h1>Two-probe architecture staircase</h1>
        <p className="deck">Train the pathway, deposit the computation.</p>
      </header>
      {children}
      <footer className="colophon">
        MIT (code) · CC BY 4.0 (data+figures) · <a href={GITHUB}>{GITHUB}</a>
        {" · "}
        concept {CONCEPT} · version pin {VERSION} (v1.5.4)
      </footer>
    </div>
  );
}
