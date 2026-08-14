import { science } from "@/lib/science";

const GITHUB = "https://github.com/PeterPonyu/architecture-staircase";

export default function RebuildPage() {
  return (
    <section className="leaf">
      <h2>Rebuild</h2>
      <p>
        Reproduce means rebuild. Clone{" "}
        <a href={GITHUB}>{GITHUB}</a>, rerun the experiment runners, and
        regenerate each staircase object.
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
      <ol className="rebuild-list">
        {science.rebuild.map((row) => (
          <li key={row.title}>{row.title}</li>
        ))}
      </ol>
    </section>
  );
}
