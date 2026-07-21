import { useState, useEffect, useRef } from "react"
import { Upload, Activity, Database, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"

// Same-origin in production (the backend serves this built SPA). Local dev points
// at the :8000 backend via frontend/.env.development. ponytail: env over hardcode.
const API = import.meta.env.VITE_API_URL ?? ""

type RiskLevel = "HIGH" | "MEDIUM" | "LOW" | "INDETERMINATE"

type Contributor = {
  feature: string
  direction: string
  value: string
  shap_impact: number
}

type Alert = {
  patient_id: string
  risk_score: number
  risk_level: RiskLevel
  aki_probability_24h: string
  top_contributors: Contributor[]
  recommended_action: string
}

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
        selected && "ring-2 ring-primary",
        className,
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

// A compact triage bar: how many patients sit at each risk level, sickest first.
function TriageSummary({ ward }: { ward: Alert[] }) {
  const counts = LEVELS.map((lvl) => ({ lvl, n: ward.filter((a) => a.risk_level === lvl).length }))
  return (
    <div className="mb-6 flex flex-wrap gap-2">
      {counts.map(({ lvl, n }) => {
        const r = RISK[lvl]
        return (
          <div key={lvl} className={cn("flex items-center gap-2 rounded-sm px-3 py-1.5", r.tint)}>
            <span className={cn("size-2 rounded-full", r.dot)} />
            <span className={cn("text-2xl font-bold tabular-nums leading-none", r.text)}>{n}</span>
            <span className={cn("text-[0.7rem] font-semibold uppercase tracking-wide", r.text)}>{lvl}</span>
          </div>
        )
      })}
    </div>
  )
}

// Parse an uploaded row into a feature object. JSON (object or [object]) or a
// CSV header+first-data-row. ponytail: naive CSV split — fine for the numeric
// MIMIC export, no quoted-comma handling.
function parseRow(filename: string, text: string): Record<string, unknown> {
  if (filename.toLowerCase().endsWith(".json")) {
    const data = JSON.parse(text)
    return Array.isArray(data) ? data[0] : data
  }
  const [header, ...rows] = text.trim().split(/\r?\n/)
  const keys = header.split(",")
  const vals = (rows[0] ?? "").split(",")
  const obj: Record<string, unknown> = {}
  keys.forEach((k, i) => {
    const v = vals[i]
    obj[k] = v === undefined || v === "" ? null : isNaN(Number(v)) ? v : Number(v)
  })
  return obj
}

// Advanced / what-if entry: a collapsed disclosure for scoring a raw patient-hour
// by hand. In production the EHR feed replaces this — it is deliberately secondary.
function AdvancedScorer() {
  const [text, setText] = useState("")
  const [scored, setScored] = useState<Alert | null>(null)
  const [err, setErr] = useState("")
  const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  async function score(payload?: string) {
    const src = payload ?? text
    setErr(""); setScored(null); setBusy(true)
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(JSON.parse(src)),
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || "Request failed")
      setScored(json as Alert)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function loadSample() {
    setErr(""); setScored(null)
    const row = await (await fetch(`${API}/sample`)).json()
    const pretty = JSON.stringify(row, null, 2)
    setText(pretty)
    score(pretty)
  }

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return
    setErr(""); setScored(null)
    try {
      const obj = parseRow(file.name, await file.text())
      const pretty = JSON.stringify(obj, null, 2)
      setText(pretty)
      score(pretty)
    } catch (err) {
      setErr(`Could not parse ${file.name}: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  return (
    <details className="mt-10 rounded-[10px] border border-border bg-card">
      <summary className="flex cursor-pointer items-center gap-2 p-4 text-sm font-semibold text-muted-foreground marker:content-['']">
        <ChevronRight className="size-4 transition-transform [details[open]_&]:rotate-90" />
        Advanced — score a raw patient-hour by hand
      </summary>
      <div className="border-t border-border p-6">
        <p className="mb-3 text-[0.8125rem] text-muted-foreground">
          For testing or what-if analysis. Load a sample, upload a patient row (.json / .csv),
          or edit the values and re-score. In production this is replaced by the EHR feed.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" onClick={loadSample}>Load sample</Button>
          <Button variant="outline" onClick={() => fileRef.current?.click()}>
            <Upload className="size-4" /> Upload row
          </Button>
          <input ref={fileRef} type="file" accept=".json,.csv" onChange={onUpload} className="hidden" />
          <Button onClick={() => score()} disabled={!text || busy}>
            {busy ? "Scoring…" : "Score"}
          </Button>
          {err && <span className="text-[0.8125rem] text-risk-high">{err}</span>}
        </div>
        {text && (
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
            className="mt-3 min-h-32 font-mono text-xs"
          />
        )}
        {scored && <AlertCard a={scored} className="mt-4 max-w-sm" />}
      </div>
    </details>
  )
}

export default function App() {
  const [ward, setWard] = useState<Alert[] | null>(null)
  const [selected, setSelected] = useState<Alert | null>(null)
  const [err, setErr] = useState("")

  useEffect(() => {
    fetch(`${API}/patients?n=12`)
      .then((r) => r.json())
      .then((data: Alert[]) => {
        const sorted = [...data].sort(
          (a, b) => SEVERITY[b.risk_level] - SEVERITY[a.risk_level] || b.risk_score - a.risk_score,
        )
        setWard(sorted)
        setSelected(sorted[0] ?? null) // focus the sickest patient by default
      })
      .catch(() => setErr("Cannot reach the inference API — is the backend running on :8000?"))
  }, [])

  return (
    <div className="min-h-svh">
      <header className="flex items-baseline gap-3 border-b bg-card px-8 py-4">
        <h1 className="flex items-center gap-2 text-xl font-semibold">
          Project Sentinel <Activity className="size-5 text-primary" />
        </h1>
        <span className="text-sm text-muted-foreground">SA-AKI early-warning ward</span>
      </header>

      <main className="mx-auto max-w-6xl p-8">
        {/* Data-source note: patients come from the connected feed, not manual entry. */}
        <div className="mb-6 flex items-start gap-2.5 rounded-sm bg-accent-soft px-4 py-2.5 text-[0.8125rem] text-muted-foreground">
          <Database className="mt-0.5 size-4 shrink-0 text-primary" />
          <span>
            Patients are loaded from the connected data source (demo: MIMIC-IV).
            In deployment this feed comes directly from the hospital EHR
            (OpenMRS · OpenClinic GA · FHIR) — clinicians assess patients with no manual entry.
          </span>
        </div>

        {err && <p className="text-[0.8125rem] text-risk-high">{err}</p>}
        {!ward && !err && <p className="text-[0.8125rem] text-muted-foreground">Loading ward…</p>}

        {ward && (
          <>
            <TriageSummary ward={ward} />

            <div className="grid gap-8 lg:grid-cols-[minmax(0,360px)_1fr]">
              {/* Focused patient — the one selected from the feed */}
              <div>
                <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Selected patient
                </p>
                {selected
                  ? <AlertCard a={selected} />
                  : <p className="text-[0.8125rem] text-muted-foreground">Select a patient from the ward →</p>}
              </div>

              {/* The ward feed — sickest first, click to focus */}
              <div>
                <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Ward · {ward.length} patients · highest risk first
                </p>
                <div className="grid gap-4 [grid-template-columns:repeat(auto-fill,minmax(300px,1fr))]">
                  {ward.map((a, i) => (
                    <AlertCard
                      key={i}
                      a={a}
                      onClick={() => setSelected(a)}
                      selected={selected?.patient_id === a.patient_id}
                    />
                  ))}
                </div>
              </div>
            </div>

            <AdvancedScorer />
          </>
        )}
      </main>
    </div>
  )
}
