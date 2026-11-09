---
name: explain-metric-change
description: Explains WHERE an additive business metric (revenue, units, number of orders) changed between two periods. Splits the change into exact, reconciled contributions by segment (region, product, channel, ...) and, when units and an item column exist, into volume, mix, price, new-item and discontinued-item effects. Use when the user asks why a number went up or down, what drove growth or decline, or how one period compares with another.
---

# Explain Metric Change

> **Implementation status (Stage 2.5)**
> - ✅ The Python tools this skill uses are implemented and tested:
>   `explain_change` (tools/change_analysis.py), `create_waterfall_chart`
>   (tools/charts.py), `parse_period` / `period_coverage` (tools/periods.py).
> - ⏳ The LLM cannot use this skill on its own yet. That needs the tool
>   registry (Stage 4), the agent loop and a skill loader (Stage 5).
>   Until then, this file is the specification of how the agent WILL behave.

## Purpose
Turn "the number changed" into "here is exactly where it changed, and what
kind of change it was", with every figure computed by Python and reconciled
to the total. The skill describes WHERE a change happened. It never claims
to know WHY.

## When to use
- "Why did revenue grow in Q2?" / "What caused the drop in March?"
- "Compare January and June sales." / "What drove the change in units sold?"
- Monthly, quarterly or yearly performance review questions.

## When NOT to use
- One period only ("What was revenue by region?") -> use `group_and_aggregate`.
- Averages or ratios (average order value, conversion rate, margin %):
  they are not additive, and the tool refuses them. Say so, and offer a
  side-by-side comparison of the two periods instead.
- No date column, or the requested periods have no rows.

## Workflow (follow in order)
1. **Inspect** with `inspect_data`. Identify:
   - the metric: a numeric column that makes sense to ADD UP (e.g. Revenue_EUR);
   - the date column;
   - for price/volume/mix: a units column, an item column (e.g. Product), and,
     if present, a unit-price column (used only for a consistency check).
2. **Clean** with `clean_data` if the date column's kind is "text".
3. **Choose the periods.** Use the user's periods. If none are given, compare
   the latest complete period with the one before it, and STATE this choice.
4. **Run** `explain_change`. Leave `dimensions` empty to let the tool choose
   segment columns automatically (2-20 groups, not ID-like), or pass the
   user's dimensions.
5. **Check** `reconciled`. If it is `false`, do NOT present the breakdown:
   report that the calculation failed its reconciliation check.
6. **Read every warning** in `warnings` before writing the answer.
7. **Visualise** with `create_waterfall_chart` (use the analysis result as input):
   - `breakdown` = the dimension from `dimension_ranking[0]`, and
   - `breakdown="price_volume_mix"` when `price_volume_mix.status == "ok"`.
8. **Answer** using the response template below.

## Tool: explain_change (input)
```json
{
  "metric": "Revenue_EUR",
  "date_column": "Date",
  "period_a": "2026-Q1",
  "period_b": "2026-Q2",
  "dimensions": null,
  "aggregate": "sum",
  "units_column": "Units",
  "item_column": "Product",
  "price_column": "Unit_Price_EUR",
  "top_n": 5
}
```
Period formats: `"2026"`, `"2026-Q1"`, `"2026-03"`, `"2026-03-15"`, or
`{"start": "2026-01-10", "end": "2026-02-20"}` (both days included).
`aggregate` is `"sum"` or `"count"` (with `"count"`, `metric` may be null).
Price/volume/mix runs only when BOTH `units_column` and `item_column` are given.

## Tool: explain_change (key output fields)
| Field | Meaning |
|---|---|
| `summary` | totals for A and B, `change`, `change_percent` (null if A is 0) |
| `drivers[dim].contributions` | top groups: `a`, `b`, `change`, `share_of_change`, `status` (existing / new / discontinued), `small_sample` |
| `drivers[dim].others` | all remaining groups added together (so the list still reconciles) |
| `drivers[dim].concentration` | 0-1: share of all movement from the single biggest group |
| `dimension_ranking` | dimensions ordered by concentration (clearest story first) |
| `top_combinations` | biggest two-dimension movements, e.g. "East / Online" (nested pairs such as Category x Product are skipped) |
| `price_volume_mix` | `status` "ok" with `effects` {volume, mix, price, new_items, discontinued_items}, or `status` "not_reliable" with `reasons` and NO numbers |
| `reconciled` | true only if EVERY breakdown adds up to the total change within tolerance |
| `warnings` | data-quality issues: small samples, missing values, new or discontinued segments, incomplete periods, negative values |
| `causation_note` | the reminder that contribution is not causation |

## The mathematics (so you can explain it correctly)
- **Contribution** of a group = its value in B - its value in A. All contributions
  add up exactly to the total change, because the metric is additive.
- **Share of change** = contribution / total change x 100. Shares can exceed 100%
  or be negative when some groups grow and others shrink.
- **Volume** = (units in B - units in A) x average price per unit in A
  -> "what selling more (or fewer) units was worth at the old average price".
- **Mix** = sum of (units_B x price_A per item) - units_B x average price in A
  -> "the effect of selling a different blend of items". Negative = shift
  toward cheaper items.
- **Price** = sum of units_B x (price_B - price_A) per item
  -> "what each item's own price change was worth".
- **New items** = B revenue of items not sold in A; **Discontinued** = minus
  A revenue of items not sold in B.
- Volume + Mix + Price + New + Discontinued = total change, checked within
  max(1e-6, 1e-9 x size of the totals).

## Interpretation rules (guardrails)
1. **Never calculate numbers yourself.** Quote only values from tool results.
2. **Contribution is not causation.** Write "East contributed -628.50", never
   "East fell because of X". Possible reasons may be offered ONLY as clearly
   labelled hypotheses ("one possibility to investigate is ...").
3. **Correlation is not causation either.** Something that happened at the same
   time (e.g. a campaign) "coincided with" the change; it is not proven to cause it.
4. **No unsupported recommendations.** Do not tell the user to cut, invest,
   reprice or close anything. You may suggest what to INVESTIGATE next.
5. **Lead with the total**, then the 2-3 biggest drivers, then the biggest
   negative driver, even when the total grew. Hidden declines matter.
6. **Respect warnings.** If groups are small samples, say results are indicative.
   If a period is incomplete, say the comparison may be unfair.
7. **Price/volume/mix "not_reliable"**: report the reasons; never estimate the
   effects yourself.
8. **Total change near zero**: do not use shares; describe offsetting movements.
9. **State every assumption**: periods, dimensions, cleaning that was applied.

## Response template
**What changed:** <metric> went from <A total> in <period A> to <B total> in
<period B> (<change>, <change_percent>%).
**Where it changed:** <2-3 largest contributions with amounts and shares>.
**Biggest drag:** <largest negative contribution, or "none">.
**Type of change:** <volume / mix / price in one sentence, or why it was not calculated>.
**Data caveats:** <warnings in plain words>.
**Not proven:** these figures show where the change happened, not why.
**Worth investigating next:** <1-2 questions, e.g. "What happened to East Online?">
[waterfall chart(s)]

## Errors and how to recover
| Error message contains | What to do |
|---|---|
| "Column(s) not found" | Re-check names with `inspect_data`; retry once |
| "not dates" | Run `clean_data`, then retry |
| "has no rows" | Tell the user the available date range; ask which periods to use |
| "overlap" | Choose non-overlapping periods |
| "not additive" | Explain; offer a side-by-side comparison instead |
| "needs BOTH units_column and item_column" | Provide both, or drop price/volume/mix |

## Limitations
- Additive metrics only (sum or count). Two periods per call.
- Describes contributions; never proves causes.
- Small datasets give indicative, not conclusive, results.
- Period completeness is judged only from the rows in the file.
- Price/volume/mix uses average realised price per item (revenue / units).
