import { useState, useEffect, useRef } from "react"
import { Upload, Activity, Database, Keyboard, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"

// Same-origin in production (the backend serves this built SPA). Local dev points
// at the :8000 backend via frontend/.env.development. ponytail: env over hardcode.
const API = import.meta.env.VITE_API_URL ?? ""

type RiskLevel = "HIGH" | "MEDIUM" | "LOW" | "INDETERMINATE"
type Mode = "ehr" | "manual"

type Contributor = { feature: string; direction: string; value: string; shap_impact: number }
type Alert = {
  patient_id: string
  risk_score: number
  risk_level: RiskLevel
  aki_probability_24h: string
  top_contributors: Contributor[]
  recommended_action: string
  features?: Record<string, number | null>  // full raw feature values (for the detail view)
}
type Config = { manual_entry: boolean; ehr_connected: boolean; ehr_url: string | null; n_features: number; features: string[] }

// Clinically meaningful raw measurements shown in the patient-detail view, grouped.
// [featureKey, human label, unit]. Derived/rolling features are omitted here — the
// detail view is the patient's clinical picture, not all 205 engineered columns.
const DETAIL_GROUPS: { title: string; items: [string, string, string][] }[] = [
  { title: "Vitals", items: [
    ["HR", "Heart rate", "bpm"], ["SBP", "Systolic BP", "mmHg"], ["DBP", "Diastolic BP", "mmHg"],
    ["MAP", "Mean arterial P.", "mmHg"], ["Resp", "Resp. rate", "/min"], ["O2Sat", "SpO₂", "%"], ["Temp", "Temp", "°C"],
  ]},
  { title: "Renal · KDIGO", items: [
    ["Creatinine", "Creatinine", "mg/dL"], ["baseline_creatinine", "Baseline creatinine", "mg/dL"],
    ["cr_above_baseline", "Δ above baseline", "mg/dL"], ["creatinine_velocity_12h", "Creatinine velocity", "/12h"],
    ["BUN", "BUN", "mg/dL"], ["urine_rate", "Urine output", "mL/kg/h"],
  ]},
  { title: "Labs", items: [
    ["Lactate", "Lactate", "mmol/L"], ["WBC", "WBC", "10³/µL"], ["Platelets", "Platelets", "10³/µL"],
    ["Hgb", "Hemoglobin", "g/dL"], ["Potassium", "Potassium", "mmol/L"], ["HCO3", "Bicarbonate", "mmol/L"],
    ["Bilirubin_total", "Bilirubin", "mg/dL"], ["pH", "pH", ""],
  ]},
  { title: "Risk factors", items: [
    ["sofa_partial", "SOFA (partial)", "pts"], ["nephrotoxin_active", "Nephrotoxin active", "0/1"],
    ["hours_since_nephrotoxin", "Since nephrotoxin", "h"], ["hours_since_sepsis", "Since sepsis onset", "h"],
  ]},
  { title: "Patient", items: [
    ["Age", "Age", "yr"], ["Gender", "Sex (1 = M)", ""], ["weight_kg", "Weight", "kg"],
  ]},
]

// Static class strings (not template-built) so Tailwind's scanner emits them.
const RISK: Record<RiskLevel, { text: string; border: string; tint: string; dot: string }> = {
  HIGH: { text: "text-risk-high", border: "border-l-risk-high", tint: "bg-risk-high-tint", dot: "bg-risk-high" },
  MEDIUM: { text: "text-risk-medium", border: "border-l-risk-medium", tint: "bg-risk-medium-tint", dot: "bg-risk-medium" },
  LOW: { text: "text-risk-low", border: "border-l-risk-low", tint: "bg-risk-low-tint", dot: "bg-risk-low" },
  INDETERMINATE: { text: "text-risk-indeterminate", border: "border-l-risk-indeterminate", tint: "bg-risk-indeterminate-tint", dot: "bg-risk-indeterminate" },
}
const LEVELS: RiskLevel[] = ["HIGH", "MEDIUM", "LOW", "INDETERMINATE"]
// Triage sort: sickest first. INDETERMINATE ranks just under HIGH — it needs a human look.
const SEVERITY: Record<RiskLevel, number> = { HIGH: 3, INDETERMINATE: 2.5, MEDIUM: 2, LOW: 1 }

function ContributorRow({ c, maxImpact }: { c: Contributor; maxImpact: number }) {
  const dir = c.direction.toLowerCase()
  const chip =
    dir === "rising" ? "bg-risk-high-tint text-risk-high"
    : dir === "falling" ? "bg-risk-low-tint text-risk-low"
    : "bg-muted text-muted-foreground"
  const arrow = dir === "rising" ? "↑" : dir === "falling" ? "↓" : "·"
  const width = maxImpact ? (Math.abs(c.shap_impact) / maxImpact) * 100 : 0
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1">
      <span className="text-sm">
        {c.feature}
        <span className={cn("ml-2 inline-flex items-center gap-1 rounded-sm px-1.5 py-px text-[0.7rem] font-semibold", chip)}>
          {arrow} {c.direction}
        </span>
      </span>
      <span className="text-sm tabular-nums text-muted-foreground">{c.value}</span>
      <div className="col-span-2 h-1 overflow-hidden rounded-full bg-border">
        <div className={cn("h-full", c.shap_impact >= 0 ? "bg-risk-high" : "bg-risk-low")} style={{ width: `${width}%` }} />
      </div>
    </div>
  )
}

function AlertCard({ a, className, onClick, selected }: {
  a: Alert; className?: string; onClick?: () => void; selected?: boolean
}) {
  const r = RISK[a.risk_level] ?? RISK.INDETERMINATE
  const maxImpact = Math.max(...a.top_contributors.map((c) => Math.abs(c.shap_impact)))
  return (
    <Card
      onClick={onClick}
      className={cn(
        "gap-0 border-l-4 p-6", r.border,
        onClick && "cursor-pointer transition-shadow hover:shadow-md",
        selected && "ring-2 ring-primary", className,
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-base font-semibold">Patient {a.patient_id}</p>
          <span className="text-xs text-muted-foreground/70">SA-AKI · 24h horizon</span>
        </div>
        <Badge variant="outline" className={cn("gap-1.5 border-transparent font-semibold tracking-wide", r.tint, r.text)}>
          <span className={cn("size-2 rounded-full", r.dot)} />
          {a.risk_level}
        </Badge>
      </div>
      <div className="my-4 flex items-baseline gap-3">
        <span className={cn("text-[2rem] font-bold leading-none tabular-nums", r.text)}>{a.risk_score}</span>
        <span className="text-sm text-muted-foreground tabular-nums">{a.aki_probability_24h} probability</span>
      </div>
      <div className="flex flex-col gap-3">
        {a.top_contributors.map((c, i) => <ContributorRow key={i} c={c} maxImpact={maxImpact} />)}
      </div>
      <div className={cn("mt-4 rounded-sm border-l-[3px] p-3 text-[0.8125rem]", r.tint, r.border)}>
        {a.recommended_action}
      </div>
    </Card>
  )
}

// Triage counts double as risk filters: click a chip to show only that level, click
// again (or the active one) to clear. `active` is the current filter (null = all).
function TriageSummary({ ward, active, onToggle }: {
  ward: Alert[]; active: RiskLevel | null; onToggle: (l: RiskLevel) => void
}) {
  return (
    <div className="mb-6 flex flex-wrap gap-2">
      {LEVELS.map((lvl) => {
        const r = RISK[lvl]
        const n = ward.filter((a) => a.risk_level === lvl).length
        return (
          <button
            key={lvl}
            onClick={() => onToggle(lvl)}
            className={cn(
              "flex items-center gap-2 rounded-sm px-3 py-1.5 transition-all hover:brightness-95",
              r.tint,
              active === lvl && "ring-2 ring-offset-1 ring-primary",
              active && active !== lvl && "opacity-45",
            )}
          >
            <span className={cn("size-2 rounded-full", r.dot)} />
            <span className={cn("text-2xl font-bold tabular-nums leading-none", r.text)}>{n}</span>
            <span className={cn("text-[0.7rem] font-semibold uppercase tracking-wide", r.text)}>{lvl}</span>
          </button>
        )
      })}
    </div>
  )
}

// Parse an uploaded row: JSON (object or [object]) or CSV header+first-data-row.
function parseRow(filename: string, text: string): Record<string, unknown> {
  if (filename.toLowerCase().endsWith(".json")) {
    const data = JSON.parse(text)
    return Array.isArray(data) ? data[0] : data
  }
  const [header, ...rows] = text.trim().split(/\r?\n/)
  const keys = header.split(","), vals = (rows[0] ?? "").split(",")
  const obj: Record<string, unknown> = {}
  keys.forEach((k, i) => {
    const v = vals[i]
    obj[k] = v === undefined || v === "" ? null : isNaN(Number(v)) ? v : Number(v)
  })
  return obj
}

// Clinical inputs a user can enter by hand — the raw, commonly-available measurements
// with full names + units. Derived features (rolling / velocity / SOFA) are omitted:
// they're computed offline or left missing, which the model tolerates.
const FORM_SECTIONS: {
  title: string
  fields: { key: string; label: string; unit?: string; select?: [string, string][] }[]
}[] = [
  { title: "Patient", fields: [
    { key: "Age", label: "Age", unit: "yr" },
    { key: "Gender", label: "Sex", select: [["1", "Male"], ["0", "Female"]] },
    { key: "weight_kg", label: "Weight", unit: "kg" },
  ]},
  { title: "Vitals", fields: [
    { key: "HR", label: "Heart rate", unit: "bpm" },
    { key: "SBP", label: "Systolic BP", unit: "mmHg" },
    { key: "DBP", label: "Diastolic BP", unit: "mmHg" },
    { key: "MAP", label: "Mean arterial pressure", unit: "mmHg" },
    { key: "Resp", label: "Respiratory rate", unit: "/min" },
    { key: "O2Sat", label: "Oxygen saturation", unit: "%" },
    { key: "Temp", label: "Temperature", unit: "°C" },
  ]},
  { title: "Renal · KDIGO", fields: [
    { key: "Creatinine", label: "Creatinine", unit: "mg/dL" },
    { key: "baseline_creatinine", label: "Baseline creatinine", unit: "mg/dL" },
    { key: "BUN", label: "Blood urea nitrogen", unit: "mg/dL" },
    { key: "urine_rate", label: "Urine output", unit: "mL/kg/h" },
  ]},
  { title: "Labs", fields: [
    { key: "Lactate", label: "Lactate", unit: "mmol/L" },
    { key: "WBC", label: "White cell count", unit: "10³/µL" },
    { key: "Platelets", label: "Platelets", unit: "10³/µL" },
    { key: "Hgb", label: "Hemoglobin", unit: "g/dL" },
    { key: "Hct", label: "Hematocrit", unit: "%" },
    { key: "Potassium", label: "Potassium", unit: "mmol/L" },
    { key: "HCO3", label: "Bicarbonate", unit: "mmol/L" },
    { key: "Glucose", label: "Glucose", unit: "mg/dL" },
    { key: "Bilirubin_total", label: "Total bilirubin", unit: "mg/dL" },
    { key: "pH", label: "Arterial pH" },
  ]},
  { title: "More labs", fields: [
    { key: "Chloride", label: "Chloride", unit: "mmol/L" },
    { key: "Calcium", label: "Calcium", unit: "mg/dL" },
    { key: "Magnesium", label: "Magnesium", unit: "mg/dL" },
    { key: "Phosphate", label: "Phosphate", unit: "mg/dL" },
    { key: "AST", label: "AST", unit: "U/L" },
    { key: "Alkalinephos", label: "Alkaline phosphatase", unit: "U/L" },
    { key: "PaCO2", label: "PaCO₂", unit: "mmHg" },
    { key: "SaO2", label: "SaO₂", unit: "%" },
    { key: "BaseExcess", label: "Base excess", unit: "mmol/L" },
    { key: "PTT", label: "PTT", unit: "s" },
    { key: "Fibrinogen", label: "Fibrinogen", unit: "mg/dL" },
  ]},
  { title: "Context / risk", fields: [
    { key: "creatinine_velocity_12h", label: "Creatinine velocity", unit: "/12h" },
    { key: "sofa_partial", label: "SOFA score (partial)", unit: "pts" },
    { key: "nephrotoxin_active", label: "Nephrotoxin given", select: [["1", "Yes"], ["0", "No"]] },
    { key: "hours_since_nephrotoxin", label: "Hours since nephrotoxin", unit: "h" },
    { key: "hours_since_sepsis", label: "Hours since sepsis onset", unit: "h" },
  ]},
]
const FORM_KEYS = FORM_SECTIONS.flatMap((s) => s.fields.map((f) => f.key))

// MANUAL mode — a friendly clinical form (not raw JSON). Enter what you have; the
// model treats the rest as missing. Load-sample / upload prefill the fields.
function ManualScorer({ onScored, allFeatures }: { onScored: (a: Alert) => void; allFeatures: string[] }) {
  const [vals, setVals] = useState<Record<string, string>>({})
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const set = (k: string, v: string) => setVals((p) => ({ ...p, [k]: v }))

  // every model feature not already in a friendly section — kept in the "advanced"
  // panel so NO field from the raw JSON is dropped (rolling stats, SOFA sub-scores,
  // missingness flags, composites…).
  const advancedKeys = allFeatures.filter((k) => !FORM_KEYS.includes(k))
  const keys = allFeatures.length ? allFeatures : FORM_KEYS

  function fillFrom(row: Record<string, unknown>) {
    const next: Record<string, string> = {}
    for (const k of keys) {
      const v = row[k]
      if (v !== null && v !== undefined && v !== "" && !Number.isNaN(Number(v))) {
        next[k] = String(Math.round(Number(v) * 1000) / 1000)  // trim float noise for display
      }
    }
    setVals(next)
  }
  async function loadSample() {
    setErr("")
    try { fillFrom(await (await fetch(`${API}/sample`)).json()) }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }
  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]; e.target.value = ""
    if (!file) return; setErr("")
    try { fillFrom(parseRow(file.name, await file.text())) }
    catch (err) { setErr(`Could not parse ${file.name}: ${err instanceof Error ? err.message : String(err)}`) }
  }
  async function score() {
    setErr(""); setBusy(true)
    const payload: Record<string, number | null> = {}
    for (const k of keys) { const v = vals[k]; payload[k] = v === undefined || v === "" ? null : Number(v) }
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || "Request failed")
      onScored(json as Alert)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  const inputCls = "h-9 w-full rounded-sm border border-border bg-background px-2.5 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
  return (
    <Card className="mb-8 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Manual entry — assess a patient</p>
          <p className="text-[0.8125rem] text-muted-foreground">Enter the values you have; leave the rest blank — the model handles missing data.</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" onClick={loadSample}>Load sample</Button>
          <Button variant="outline" onClick={() => fileRef.current?.click()}><Upload className="size-4" /> Upload</Button>
          <input ref={fileRef} type="file" accept=".json,.csv" onChange={onUpload} className="hidden" />
        </div>
      </div>

      <div className="mt-5 grid gap-x-8 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">
        {FORM_SECTIONS.map((s) => (
          <div key={s.title}>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{s.title}</p>
            <div className="flex flex-col gap-2.5">
              {s.fields.map((f) => (
                <label key={f.key} className="flex flex-col gap-1 text-[0.8125rem] text-muted-foreground">
                  <span>{f.label}{f.unit && <span className="text-muted-foreground/60"> ({f.unit})</span>}</span>
                  {f.select
                    ? <select className={inputCls} value={vals[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)}>
                        <option value="">—</option>
                        {f.select.map(([v, lab]) => <option key={v} value={v}>{lab}</option>)}
                      </select>
                    : <input type="number" step="any" className={inputCls} placeholder="—"
                        value={vals[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)} />}
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {advancedKeys.length > 0 && (
        <details className="mt-6 rounded-[10px] border border-border">
          <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-muted-foreground">
            Advanced — all other features ({advancedKeys.length}) · rolling windows, SOFA sub-scores, missingness flags, composites
          </summary>
          <div className="grid gap-x-6 gap-y-3 border-t border-border p-5 sm:grid-cols-3 lg:grid-cols-4">
            {advancedKeys.map((k) => (
              <label key={k} className="flex flex-col gap-1 text-xs text-muted-foreground">
                <span className="truncate font-mono" title={k}>{k}</span>
                <input type="number" step="any" className={inputCls} placeholder="—"
                  value={vals[k] ?? ""} onChange={(e) => set(k, e.target.value)} />
              </label>
            ))}
          </div>
        </details>
      )}

      <div className="mt-5 flex items-center gap-3">
        <Button onClick={score} disabled={busy}>{busy ? "Scoring…" : "Assess patient"}</Button>
        {err && <span className="text-[0.8125rem] text-risk-high">{err}</span>}
      </div>
    </Card>
  )
}

// EHR mode — pull a patient from the connected FHIR feed by id. No typing.
function EhrPanel({ config, onScored }: { config: Config; onScored: (a: Alert) => void }) {
  const [pid, setPid] = useState(""); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false)

  async function fetchPatient() {
    setErr(""); setBusy(true)
    try {
      const res = await fetch(`${API}/ehr/${encodeURIComponent(pid)}`)
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || "Request failed")
      onScored(json as Alert)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  if (!config.ehr_connected) {
    return (
      <Card className="mb-8 gap-2 p-6">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Live EHR feed — not connected</p>
        <p className="text-[0.8125rem] text-muted-foreground">
          Set <code className="rounded-sm bg-muted px-1 py-px">EHR_FHIR_URL</code> to the hospital's FHIR R4 base
          (<strong>OpenClinic GA</strong>'s FHIR endpoint at the target site, or any FHIR server) to fetch
          patients directly from the record — no manual entry. The ward below is served from the demo data
          source meanwhile.
        </p>
      </Card>
    )
  }
  return (
    <Card className="mb-8 p-6">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Live EHR feed — assess by patient id</p>
      <p className="text-[0.8125rem] text-muted-foreground">
        Connected to <code className="rounded-sm bg-muted px-1 py-px">{config.ehr_url}</code>. Enter a patient id;
        the record is fetched from the EHR and scored — no manual entry.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <input value={pid} onChange={(e) => setPid(e.target.value)} placeholder="FHIR patient id"
          onKeyDown={(e) => e.key === "Enter" && pid && fetchPatient()}
          className="h-9 max-w-56 rounded-sm border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary" />
        <Button onClick={fetchPatient} disabled={!pid || busy}>{busy ? "Fetching…" : "Fetch & assess"}</Button>
        {err && <span className="text-[0.8125rem] text-risk-high">{err}</span>}
      </div>
    </Card>
  )
}

function ModeToggle({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  const opt = (m: Mode, icon: React.ReactNode, label: string) => (
    <button
      onClick={() => setMode(m)}
      className={cn(
        "flex items-center gap-2 rounded-sm px-3 py-1.5 text-sm font-medium transition-colors",
        mode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
      )}
    >
      {icon} {label}
    </button>
  )
  return (
    <div className="inline-flex rounded-md border border-border bg-card p-0.5">
      {opt("ehr", <Database className="size-4" />, "Live EHR feed")}
      {opt("manual", <Keyboard className="size-4" />, "Manual entry")}
    </div>
  )
}

// Full patient detail — opens on card click. Shows the risk header, every SHAP
// contributor, and the patient's grouped clinical values (not just the card's top-3).
function PatientDetail({ a, onClose }: { a: Alert; onClose: () => void }) {
  const r = RISK[a.risk_level] ?? RISK.INDETERMINATE
  const maxImpact = Math.max(...a.top_contributors.map((c) => Math.abs(c.shap_impact)), 1)
  const f = a.features ?? {}
  const fmt = (v: number | null | undefined) =>
    v === null || v === undefined ? "—" : Number.isInteger(v) ? String(v) : v.toFixed(2)
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 p-4 sm:p-8"
      onClick={onClose}>
      <Card className={cn("w-full max-w-3xl gap-0 border-l-4 p-6", r.border)} onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-lg font-semibold">Patient {a.patient_id}</p>
            <span className="text-xs text-muted-foreground/70">Sepsis-Associated AKI · 24h horizon</span>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className={cn("gap-1.5 border-transparent font-semibold tracking-wide", r.tint, r.text)}>
              <span className={cn("size-2 rounded-full", r.dot)} /> {a.risk_level}
            </Badge>
            <button onClick={onClose} className="rounded-sm p-1 text-muted-foreground hover:bg-muted" aria-label="Close">
              <X className="size-5" />
            </button>
          </div>
        </div>

        <div className="my-4 flex items-baseline gap-3">
          <span className={cn("text-[2.5rem] font-bold leading-none tabular-nums", r.text)}>{a.risk_score}</span>
          <span className="text-sm text-muted-foreground tabular-nums">{a.aki_probability_24h} probability of AKI</span>
        </div>

        {/* clinical values, grouped */}
        <div className="grid gap-x-8 gap-y-5 sm:grid-cols-2">
          {DETAIL_GROUPS.map((g) => (
            <div key={g.title}>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{g.title}</p>
              <div className="flex flex-col gap-1">
                {g.items.map(([key, label, unit]) => (
                  <div key={key} className="flex items-baseline justify-between gap-3 border-b border-border/60 py-1 text-sm">
                    <span className="text-muted-foreground">{label}</span>
                    <span className={cn("tabular-nums", f[key] == null && "text-muted-foreground/50")}>
                      {fmt(f[key])} <span className="text-xs text-muted-foreground/60">{unit}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* full SHAP explanation */}
        <p className="mt-6 mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Why this risk — model factors (SHAP)
        </p>
        <div className="flex flex-col gap-3">
          {a.top_contributors.map((c, i) => <ContributorRow key={i} c={c} maxImpact={maxImpact} />)}
        </div>

        <div className={cn("mt-5 rounded-sm border-l-[3px] p-3 text-[0.8125rem]", r.tint, r.border)}>
          <span className="font-semibold">Recommended action: </span>{a.recommended_action}
        </div>
      </Card>
    </div>
  )
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [mode, setMode] = useState<Mode>("ehr")
  const [ward, setWard] = useState<Alert[] | null>(null)
  const [selected, setSelected] = useState<Alert | null>(null)
  const [detail, setDetail] = useState<Alert | null>(null)  // full-detail modal
  const [search, setSearch] = useState("")                  // filter by patient id
  const [riskFilter, setRiskFilter] = useState<RiskLevel | null>(null)
  const [err, setErr] = useState("")

  useEffect(() => {
    fetch(`${API}/config`).then((r) => r.json()).then(setConfig).catch(() => {})
    fetch(`${API}/patients?n=12`)
      .then((r) => r.json())
      .then((data: Alert[]) => {
        const sorted = [...data].sort(
          (a, b) => SEVERITY[b.risk_level] - SEVERITY[a.risk_level] || b.risk_score - a.risk_score,
        )
        setWard(sorted); setSelected(sorted[0] ?? null)
      })
      .catch(() => setErr("Cannot reach the inference API — is the backend running on :8000?"))
  }, [])

  return (
    <div className="min-h-svh">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b bg-card px-8 py-4">
        <div className="flex items-baseline gap-3">
          <h1 className="flex items-center gap-2 text-xl font-semibold">
            Project Sentinel <Activity className="size-5 text-primary" />
          </h1>
          <span className="text-sm text-muted-foreground">SA-AKI early-warning ward</span>
        </div>
        <ModeToggle mode={mode} setMode={setMode} />
      </header>

      <main className="mx-auto max-w-6xl p-8">
        {mode === "ehr" && (
          <div className="mb-6 flex items-start gap-2.5 rounded-sm bg-accent-soft px-4 py-2.5 text-[0.8125rem] text-muted-foreground">
            <Database className="mt-0.5 size-4 shrink-0 text-primary" />
            <span>
              Patients come from the connected data source (demo: MIMIC-IV). In deployment this feed is the
              hospital HIS — <strong>OpenClinic GA</strong> (FHIR API) at the target site, or any FHIR R4 /
              OpenMRS server — so clinicians assess patients with no manual entry.
            </span>
          </div>
        )}

        {config && (mode === "manual"
          ? <ManualScorer allFeatures={config.features ?? []} onScored={(a) => { setSelected(a); setDetail(a) }} />
          : <EhrPanel config={config} onScored={(a) => { setSelected(a); setDetail(a) }} />)}

        {err && <p className="text-[0.8125rem] text-risk-high">{err}</p>}
        {!ward && !err && <p className="text-[0.8125rem] text-muted-foreground">Loading ward…</p>}

        {ward && (() => {
          const q = search.trim().toLowerCase()
          const filtered = ward.filter(
            (a) => (!riskFilter || a.risk_level === riskFilter) && a.patient_id.toLowerCase().includes(q),
          )
          return (
          <>
            <TriageSummary
              ward={ward}
              active={riskFilter}
              onToggle={(l) => setRiskFilter((cur) => (cur === l ? null : l))}
            />
            <div className="grid gap-8 lg:grid-cols-[minmax(0,360px)_1fr]">
              <div>
                <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Selected patient <span className="normal-case text-muted-foreground/60">· click for full detail</span>
                </p>
                {selected
                  ? <AlertCard a={selected} onClick={() => setDetail(selected)} />
                  : <p className="text-[0.8125rem] text-muted-foreground">Select a patient from the ward →</p>}
              </div>
              <div>
                <div className="mb-3 flex flex-wrap items-center gap-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Ward · {filtered.length}{filtered.length !== ward.length && ` of ${ward.length}`} patients
                    {!riskFilter && !q && " · highest risk first"}
                  </p>
                  <input
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search patient id…"
                    className="ml-auto h-8 w-44 rounded-sm border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
                  />
                  {(riskFilter || q) && (
                    <button onClick={() => { setRiskFilter(null); setSearch("") }}
                      className="text-xs font-medium text-primary hover:underline">clear</button>
                  )}
                </div>
                <div className="grid gap-4 [grid-template-columns:repeat(auto-fill,minmax(300px,1fr))]">
                  {filtered.length === 0 && (
                    <p className="text-[0.8125rem] text-muted-foreground">No patients match.</p>
                  )}
                  {filtered.map((a, i) => (
                    <AlertCard key={i} a={a} onClick={() => { setSelected(a); setDetail(a) }}
                      selected={selected?.patient_id === a.patient_id} />
                  ))}
                </div>
              </div>
            </div>
          </>
          )
        })()}
      </main>

      {detail && <PatientDetail a={detail} onClose={() => setDetail(null)} />}
    </div>
  )
}
