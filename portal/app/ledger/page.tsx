"use client";

import { useEffect, useState } from "react";

type ObjectRow = {
  id?: string;
  generator?: string;
};

type ObjectIndex = {
  figures?: ObjectRow[];
};

export default function LedgerPage() {
  const [rows, setRows] = useState<ObjectRow[]>([]);

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
          return response.json() as Promise<ObjectIndex>;
        })
        .then((index) => {
          if (cancelled) {
            return;
          }
          const objects = (index.figures || []).map((row) => ({
            id: row.id,
            generator: row.generator,
          }));
          setRows(objects);
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
      <p>Named staircase objects and the generators that rebuild them.</p>
      <table className="name-index">
        <thead>
          <tr>
            <th>Object</th>
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
