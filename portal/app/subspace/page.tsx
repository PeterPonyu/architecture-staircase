import { Entries } from "@/components/Entries";
import { science } from "@/lib/science";

export default function SubspacePage() {
  return (
    <section className="leaf">
      <h2>Subspace</h2>
      <p>
        The trained update occupies a localized induction-head subspace.
        Context length changes the subspace, not the probe split.
      </p>
      <Entries entries={science.subspace} />
    </section>
  );
}
