// Short hover explanations grounded in the Kineticolor paper.
// Each: how it's calculated + how to read it.
export interface MetricInfo { key: string; label: string; how: string; read: string; }

export const DELTA_E_INFO: MetricInfo = {
  key: "normalized_delta_e",
  label: "ΔE (color change)",
  how: "ΔE is the straight-line (Euclidean) distance in CIE-L*a*b* color space between each frame's average color and the reference frame (t=0). Shown normalized 0–1 (divided by its maximum).",
  read: "Rises as the mixture's color changes, then flattens (plateaus) once color stops changing — i.e. mixing is complete. The 0.90 / 0.95 / 0.99 markers are the first times ΔE reaches 90 / 95 / 99% of its final value. ΔE perception: <1 invisible, 2–10 visible at a glance, ~100 opposite colors.",
};

export const METRIC_INFO: Record<string, MetricInfo> = {
  contact_perimeter: {
    key: "contact_perimeter", label: "Contact",
    how: "Each frame is made black/white by a grayscale threshold; Contact is the total perimeter between black and white regions.",
    read: "Peaks during the mixing transition (mixed and unmixed regions coexist) and decays toward ~0 as the vessel becomes uniform.",
  },
  contrast: {
    key: "contrast", label: "Contrast (GLCM)",
    how: "Computed from the Gray-Level Co-occurrence Matrix: the gray-level difference across neighbouring pixel pairs.",
    read: "Highest when the image is most visibly heterogeneous; falls toward zero as the mixture becomes homogeneous.",
  },
  homogeneity: {
    key: "homogeneity", label: "Homogeneity (GLCM)",
    how: "From the GLCM: how close the matrix is to diagonal (neighbouring pixels sharing the same gray level).",
    read: "Increases as pixels become similar — higher means more thoroughly mixed.",
  },
  energy: {
    key: "energy", label: "Energy / ASM (GLCM)",
    how: "Angular Second Moment from the GLCM — the amount of single 'block' color (sum of squared probabilities).",
    read: "Low for noisy/heterogeneous frames, rises toward its maximum (1) as the frame becomes one uniform color.",
  },
  variance_delta_e: {
    key: "variance_delta_e", label: "Variance (by cell)",
    how: "The region is split into a 5×5 grid; this is the variance of average ΔE across those cells.",
    read: "High when different areas of the vessel differ in color; drops as mixing evens them out.",
  },
};
