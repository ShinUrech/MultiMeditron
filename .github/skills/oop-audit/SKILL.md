---
name: oop-audit
description: 'Audit a codebase for OOP principle violations. Use for: reviewing SOLID principles, encapsulation, inheritance abuse, cohesion/coupling issues, God classes, feature envy, anemic domain models, or when asked to review OOP design. Triggers: "audit OOP", "check SOLID", "review class design", "OOP principles", "refactor classes".'
argument-hint: 'Optional: specify a module, folder, or class name to scope the audit'
---

# OOP Audit Skill

Systematically audit a codebase (or a scoped portion of it) for violations of Object-Oriented Programming principles, produce a prioritized findings report, and suggest concrete fixes.

## Principles Checked

| Category | Principles |
|----------|-----------|
| **SOLID** | Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion |
| **Classic OOP** | Encapsulation, Abstraction, Inheritance vs Composition |
| **Design Smells** | God Class, Feature Envy, Data Class, Shotgun Surgery, Divergent Change, Refused Bequest |
| **Coupling/Cohesion** | High coupling between modules, low cohesion within classes |

---

## Procedure

### Step 1 — Scope the Audit

If the user provided an argument (module, folder, class), limit the scope. Otherwise, discover entry points:

```
list the top-level source directories
identify the primary language and class/module structure
```

Start from the most central modules (most imported / highest fan-in). Limit initial pass to ≤ 20 files unless user asks for full scan.

### Step 2 — Collect Structural Signals

For each file/class in scope, look for:

- **Class size**: >300 lines or >10 public methods → SRP suspect
- **Method length**: >40 lines → likely doing too much
- **Constructor parameter count**: >5 → possible violation of DIP or SRP
- **Inheritance depth**: >3 levels → inheritance abuse risk
- **Direct instantiation of concrete types** inside a class (e.g. `obj = ConcreteClass()`) → DIP violation
- **Public fields / mutable state exposed directly** → encapsulation violation
- **Subclass that overrides almost everything** or raises `NotImplementedError` in parent → LSP suspect
- **Static methods operating on external data** → Feature Envy or misplaced responsibility
- **Classes with only getters/setters and no behavior** → Anemic Domain Model
- **`isinstance` checks in business logic** → OCP/LSP violation

### Step 3 — Classify Findings

For each finding, assign:

| Field | Values |
|-------|--------|
| **Principle** | SRP / OCP / LSP / ISP / DIP / Encapsulation / Cohesion / Coupling / Inheritance |
| **Severity** | Critical / Major / Minor |
| **Location** | File + class/method + line range |
| **Evidence** | Quoted code snippet (≤5 lines) |
| **Fix** | Concrete suggestion (extract class, inject dependency, use composition, etc.) |

**Severity guide:**
- **Critical**: Architectural issue that makes the code resistant to change or testing
- **Major**: Clear violation that increases coupling or reduces cohesion significantly
- **Minor**: Style issue or debatable design choice

### Step 4 — Produce the Report

Output a structured Markdown report:

```
## OOP Audit Report — <scope>

### Summary
- Files scanned: N
- Findings: X critical, Y major, Z minor

### Critical Findings
#### [C1] <Short title>
- **Principle violated**: ...
- **Location**: `path/to/file.py` line N–M
- **Evidence**: ...
- **Suggested fix**: ...

### Major Findings
...

### Minor Findings
...

### Positive Observations
(Note any well-designed patterns found — good abstractions, proper DI, etc.)
```

### Step 5 — Offer to Fix

After the report, ask:
> "Would you like me to fix any of these findings? I can refactor them one at a time, starting with the highest severity."

If user agrees, apply fixes incrementally — one class/method at a time — and re-read the changed file after each edit to validate the fix didn't introduce new violations.

---

## Decision Points

**Whole repo vs focused scope?**  
→ If no argument given and repo is large (>50 source files), audit the 5–10 most central modules first, then ask if user wants to expand.

**Language without classes (e.g. functional Python)?**  
→ Shift focus to module cohesion, function responsibilities, global state, and dependency injection patterns instead of class-level checks.

**Disagreement on severity?**  
→ Explain the trade-off (e.g. "this anemic model is intentional if using a DTO pattern") and let the user override.

---

## Quality Criteria / Done Checklist

- [ ] All files in scope have been read (not just grepped)
- [ ] Each finding includes location + evidence + fix
- [ ] Findings are sorted by severity (Critical first)
- [ ] At least one positive observation noted (avoid purely negative reports)
- [ ] User offered the option to apply fixes

---

## OOP Quick Reference

**SRP** — A class should have one reason to change. One responsibility = one axis of change.  
**OCP** — Open for extension, closed for modification. Add behavior via new classes, not by editing existing ones.  
**LSP** — Subtypes must be substitutable for their base type without altering correctness.  
**ISP** — Prefer many small, specific interfaces over one large generic one.  
**DIP** — Depend on abstractions, not concretions. High-level modules should not import low-level ones directly.  
**Encapsulation** — Hide internal state; expose only what callers need.  
**Composition over Inheritance** — Prefer assembling behavior from small objects over deep inheritance chains.
