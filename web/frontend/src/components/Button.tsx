import React from "react";
import { Spinner } from "./Spinner";

type Variant = "primary" | "secondary" | "ghost";
const base: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
  font: "inherit", fontSize: 14, fontWeight: 600, lineHeight: 1,
  padding: "9px 16px", borderRadius: "var(--kc-radius-sm)", cursor: "pointer",
  border: "1px solid transparent", transition: "transform 150ms var(--ease-out), background-color 150ms ease, opacity 150ms ease",
};
const variants: Record<Variant, React.CSSProperties> = {
  primary: { background: "var(--kc-accent)", color: "#fff" },
  secondary: { background: "var(--kc-surface)", color: "var(--kc-text)", borderColor: "var(--kc-border)" },
  ghost: { background: "transparent", color: "var(--kc-muted)" },
};

export function Button({ variant = "primary", loading, disabled, children, style, ...rest }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }) {
  const isDisabled = disabled || loading;
  return (
    <button {...rest} disabled={isDisabled} data-kc-btn data-variant={variant}
      style={{ ...base, ...variants[variant], opacity: isDisabled ? 0.55 : 1,
               cursor: isDisabled ? "default" : "pointer", ...style }}>
      {loading && <Spinner size={15} color={variant === "primary" ? "rgba(255,255,255,.85)" : undefined} />}
      {children}
    </button>
  );
}
