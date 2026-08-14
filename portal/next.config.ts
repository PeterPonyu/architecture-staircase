import path from "node:path";
import type { NextConfig } from "next";

const repo = "architecture-staircase";

const nextConfig: NextConfig = {
  output: "export",
  outputFileTracingRoot: path.join(__dirname),
  basePath: `/${repo}`,
  assetPrefix: `/${repo}`,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
