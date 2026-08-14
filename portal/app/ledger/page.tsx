"use client";

import { useEffect, useState } from "react";

type FigureRow = {
  id?: string;
  generator?: string;
};

type FigureIndex = {
  pipeline?: string;
  figures?: FigureRow[];
};

export default function LedgerPage() {
  const [rows, setRows] = useState<FigureRow[]>([]);
  const [pipeline, setPipeline] = useState("figs/PIPELINE.md");

  useEffect(() => {
    const candidates = ["/architecture-staircase/data/figures.json"];
    let cancelled = false;
    function load(i: number) {
      if (cancelled || i >= candidates.length) {
        return;
      }
      fetch(candidates[i])
        .then((response) => {
          if (!response.ok) {
            throw new Error("missing");
          }
          return response.json() as Promise<FigureIndex>;
        })
        .then((index) => {
          if (cancelled) {
            return;
          }
          setPipeline(index.pipeline || "figs/PIPELINE.md");
          const figures = (index.figures || []).map((fig) => ({
            id: fig.id,
            generator: fig.generator,
          }));
          setRows(figures);
        })
        .catch(() => {
          load(i + 1);
        });
    }
    load(0);
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="leaf">
      <h2>Ledger</h2>
      <p>
        Name index from <code>papers/FIGURE-INDEX.json</code> (copied at build to
        <code> data/figures.json</code>). Warehouse ids and generators only.
        Pipeline: <code>{pipeline}</code>.
      </p>
      <table className="name-index">
        <thead>
          <tr>
            <th>Warehouse id</th>
            <th>Generator</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>{row.generator}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
