"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Probes" },
  { href: "/staircase/", label: "Staircase" },
  { href: "/scale/", label: "Scale" },
  { href: "/subspace/", label: "Subspace" },
  { href: "/ledger/", label: "Ledger" },
  { href: "/reproduce/", label: "Rebuild" },
] as const;

function isActive(pathname: string, href: string): boolean {
  if (href === "/") {
    return pathname === "/" || pathname === "";
  }
  return pathname === href || pathname === href.replace(/\/$/, "");
}

export function Spine() {
  const pathname = usePathname() ?? "/";
  return (
    <nav className="spine" aria-label="Staircase spine">
      {NAV.map((item) => {
        const active = isActive(pathname, item.href);
        return (
          <Link
            key={item.href}
            className={active ? "spine-link is-active" : "spine-link"}
            href={item.href}
            aria-current={active ? "page" : undefined}
          >
            <span className="spine-mark" aria-hidden="true" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
