import { Entries } from "@/components/Entries";
import { science } from "@/lib/science";

export default function ScalePage() {
  return (
    <section className="leaf">
      <h2>Scale</h2>
      <p>
        Width and depth move when the staircase locks. The two probes stay
        separate across the scale grid.
      </p>
      <Entries entries={science.scale} />
    </section>
  );
}
