import React from "react";

export function Spinner({ size = 18, color }: { size?: number; color?: string }) {
  const bars = 12;
  return (
    <span className="kc-spinner" role="status" aria-label="Loading"
          style={{ width: size, height: size, color }}>
      {Array.from({ length: bars }).map((_, i) => (
        <i key={i} style={{
          transform: `rotate(${i * (360 / bars)}deg)`,
          animationDelay: `${(i / bars - 1).toFixed(3)}s`,
        }} />
      ))}
    </span>
  );
}
