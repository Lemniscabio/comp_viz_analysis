import { createContext, useContext } from "react";
import type { Me } from "./api";
export const MeContext = createContext<Me | null>(null);
export const useMe = () => useContext(MeContext);
export const isAdmin = (me: Me | null) => me?.role === "admin" && me?.status === "active";
export const canRun = (me: Me | null) => me?.status === "active" && (me?.role === "runner" || me?.role === "admin");
