import raw from "../public/science.json";

export type ScienceEntry = {
  title: string;
  panels: string[];
  questions: string[];
};

export type Science = {
  probes: {
    freeze: ScienceEntry[];
    ablation: ScienceEntry[];
  };
  staircase: ScienceEntry[];
  scale: ScienceEntry[];
  subspace: ScienceEntry[];
  ledger: ScienceEntry[];
  rebuild: { title: string }[];
};

export const science = raw as Science;
