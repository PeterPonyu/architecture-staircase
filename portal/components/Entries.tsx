import type { ScienceEntry } from "@/lib/science";

export function Entries({
  entries,
  empty,
}: {
  entries: ScienceEntry[];
  empty?: string;
}) {
  if (!entries.length) {
    return empty ? <p className="door">{empty}</p> : null;
  }
  return (
    <div className="entries">
      {entries.map((entry) => (
        <article className="entry" key={entry.title}>
          <h3>{entry.title}</h3>
          {entry.panels.length > 0 ? (
            <p className="entry-panels">{entry.panels.join(" · ")}</p>
          ) : null}
          {entry.questions.length > 0 ? (
            <ol className="entry-questions">
              {entry.questions.map((question) => (
                <li key={question}>{question}</li>
              ))}
            </ol>
          ) : null}
        </article>
      ))}
    </div>
  );
}
