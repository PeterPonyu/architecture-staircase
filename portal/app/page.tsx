import { Entries } from "@/components/Entries";
import { science } from "@/lib/science";

export default function ProbesPage() {
  return (
    <div className="spread" id="spread-probes">
      <section className="page page-freeze" aria-labelledby="freeze-title">
        <h2 id="freeze-title">Training necessity (freeze)</h2>
        <p className="door">
          The freeze probe asks which modules must be trained for the degree
          staircase to appear. Necessity lives on this page.
        </p>
        <Entries entries={science.probes.freeze} />
      </section>
      <aside className="gutter" aria-label="Dissociation spine">
        <span className="stitch" aria-hidden="true" />
        <div className="chip">SI</div>
        <span className="stitch" aria-hidden="true" />
        <div className="chip">TOST</div>
        <span className="stitch" aria-hidden="true" />
        <div className="chip">Holm</div>
        <span className="stitch stitch-long" aria-hidden="true" />
        <div className="chip chip-bottom">clock ≠ compass</div>
      </aside>
      <section className="page page-ablation" aria-labelledby="ablation-title">
        <h2 id="ablation-title">Trained computation (ablation)</h2>
        <p className="door">
          The ablation probe asks which trained modules carry the computation
          once the staircase is in place. Ownership lives on this page.
        </p>
        <Entries entries={science.probes.ablation} />
      </section>
    </div>
  );
}
