export default function ProbesPage() {
  return (
    <div className="spread" id="spread-probes">
      <section className="page page-freeze" aria-labelledby="freeze-title">
        <h2 id="freeze-title">Training necessity (freeze)</h2>
        <p className="door">
          Left page of the facing-page grammar. This leaf is a door to the freeze
          probe in <code>papers/C/main.tex</code>, not a results board.
        </p>
        <div className="ruled-slot" aria-hidden="true" />
        <div className="ruled-slot" aria-hidden="true" />
        <div className="ruled-slot" aria-hidden="true" />
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
          Right page of the facing-page grammar. This leaf is a door to the
          ablation probe in <code>papers/C/main.tex</code>, not a results board.
        </p>
        <div className="ruled-slot" aria-hidden="true" />
        <div className="ruled-slot" aria-hidden="true" />
      </section>
    </div>
  );
}
