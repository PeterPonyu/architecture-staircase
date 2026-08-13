import type { Metadata } from "next";
import { Literata, STIX_Two_Text } from "next/font/google";
import { Notebook } from "@/components/Notebook";
import "./globals.css";

const literata = Literata({
  subsets: ["latin"],
  variable: "--font-literata",
  display: "swap",
  axes: ["opsz"],
});

const stix = STIX_Two_Text({
  subsets: ["latin"],
  variable: "--font-stix",
  weight: ["400", "600"],
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
