# Technical Debt Tracker

Known debt, its impact, and remediation plans.
Updated by `/critique` after implementation reviews.

## Severity Key

| Level | Meaning |
|-------|---------|
| `high` | Blocks future work or causes frequent friction |
| `medium` | Annoying but workable, should fix in next related change |
| `low` | Cosmetic or theoretical, fix opportunistically |

---

## DEBT-001: Incomplete MLX Type Stubs

**Severity:** low
**Filed:** 2026-03-11 (migrated from MEMORY.md)
**Status:** open

**Description:** ~116 mypy errors from incomplete MLX type stubs,
not real bugs. Suppressed with type: ignore comments in some places.

**Impact:** Noisy mypy output obscures real type errors.
**Remediation:** Contribute upstream stubs or maintain local stub
overrides. Low priority since errors are all false positives.

---

## DEBT-002: MoE Constructor Argument Mismatch

**Severity:** medium
**Filed:** 2026-03-11 (migrated from devlog)
**Status:** resolved

**Description:** MoE FFNs needed special handling because their
constructors take extra arguments (`n_experts`, `top_k`) not in
the standard FFN interface.

**Resolution:** `BlockConfig` carries MoE fields; MoE constructors
read from config with optional overrides. Clean but adds fields to
BlockConfig that only MoE uses.

**Residual debt:** BlockConfig has MoE-specific fields that are
`None` for non-MoE configurations. Could be cleaner with a
separate MoEConfig, but the current approach works.

---

*Add new debt items via `/critique` or manually.*
