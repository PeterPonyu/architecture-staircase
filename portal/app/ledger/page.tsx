import { science } from "@/lib/science";

export default function LedgerPage() {
  return (
    <section className="leaf">
      <h2>Ledger</h2>
      <p>Named staircase objects and the questions each one holds.</p>
      <table className="name-index">
        <thead>
          <tr>
            <th>Object</th>
            <th>Holds</th>
          </tr>
        </thead>
        <tbody>
          {science.ledger.map((row) => (
            <tr key={row.title}>
              <td>{row.title}</td>
              <td>
                {row.panels.length
                  ? row.panels.join(" · ")
                  : row.questions[0] || "Geometry leaf"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
