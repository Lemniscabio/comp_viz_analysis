import React from "react";

export function Skeleton({ width = "100%", height = 14, radius, style }: {
  width?: number | string; height?: number | string; radius?: number | string; style?: React.CSSProperties;
}) {
  return <div className="kc-skel" style={{ width, height, borderRadius: radius, ...style }} />;
}

export function SkeletonRows({ rows = 3, height = 16, gap = 10 }: { rows?: number; height?: number; gap?: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap }}>
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} width={`${90 - (i % 3) * 12}%`} height={height} />
      ))}
    </div>
  );
}
