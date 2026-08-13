const GITHUB = "https://github.com/PeterPonyu/architecture-staircase";

export default function ReproducePage() {
  return (
    <section className="leaf">
      <h2>Reproduce</h2>
      <p>
        Clone <a href={GITHUB}>{GITHUB}</a>. Pointer manuscript:
        <code> papers/C/main.tex</code> with
        <code>{"\\input{../figs/figpreamble.tex}"}</code>.
      </p>
      <p>
        Concept DOI{" "}
        <a href="https://doi.org/10.5281/zenodo.21020348">
          10.5281/zenodo.21020348
        </a>
        . Version pin v1.5.4:{" "}
        <a href="https://doi.org/10.5281/zenodo.21882597">
          10.5281/zenodo.21882597
        </a>
        .
      </p>
      <p>
        Rebuild via <code>papers/figs/PIPELINE.md</code>. Compiled
        <code> tex/</code> and <code>vec/</code> tiers are gitignored and are not
        hosted here.
      </p>
    </section>
  );
}
