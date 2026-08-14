import type { Metadata } from "next";
import localFont from "next/font/local";
import { Notebook } from "@/components/Notebook";
import "./globals.css";

// Self-hosted type pair (SIL OFL 1.1 — see app/fonts/OFL-*.txt). Variable
// woff2 files are committed; the build fetches no fonts. Literata ships its
// full opsz + wght axes; STIX Two Text ships wght 400-700.
const literata = localFont({
  src: "./fonts/literata-latin-opsz-normal.woff2",
  variable: "--font-literata",
  display: "swap",
});

const stix = localFont({
  src: "./fonts/stix-two-text-latin-wght-normal.woff2",
  variable: "--font-stix",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Lab book C · architecture-staircase",
  description: "Two-probe lab notebook door for architecture-staircase.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${literata.variable} ${stix.variable}`}>
        <Notebook>{children}</Notebook>
      </body>
    </html>
  );
}
