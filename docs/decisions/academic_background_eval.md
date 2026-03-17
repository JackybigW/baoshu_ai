# Academic Background Field Evaluation

## Background

An interviewer test exposed a current constraint: the system only routes into the
"academic_background is still too vague, continue asking" branch when the profile
still has missing fields.

## Current Problem

`academic_background: Optional[str]` is intentionally broad.

Advantages:

- flexible enough for messy user inputs

Tradeoffs:

1. information density is unstable, for example only `本科在读，商科方向`
2. the interviewer needs extra regex logic to decide whether grades are already present
3. downstream consultant and product planning logic cannot directly rely on structured signals

## Why We Are Not Splitting the Field Yet

Immediately splitting into `current_school`, `gpa`, `score_band`, and similar
sub-fields would create obvious costs:

1. higher extractor burden and more structured-output instability
2. user utterances are often incomplete, so hard-splitting would create many empty fields
3. interviewer follow-up chains would become longer and less natural
4. current routing and product filtering do not yet strongly depend on those finer-grained fields

## Current Recommendation

Keep `academic_background` as a broad field for now.

In the short term, the current approach is acceptable:

- broad extractor field
- interviewer asks for clarification only when needed

## Future Evaluation Directions

Prefer staged refinement instead of a full schema split.

### Option A: Semi-structured

- keep `academic_background`
- add one low-risk field such as `score_snapshot`
- only extract the highest-signal score expressions: GPA, average score, predicted A-Level, IELTS, TOEFL

### Option B: Two-stage extraction

- stage 1: extractor still returns the broad field
- stage 2: only parse additional structure when consultant logic actually needs it

## Questions to Watch

1. which non-score numbers are incorrectly treated as score evidence, such as `211`, `985`, or `Top50`
2. whether users tolerate interviewer follow-up questions about grades
3. whether the broad field is already sufficient for initial consultant output

## Conclusion

Do not change the schema yet.

Keep `academic_background` broad until the current merge logic and prompt behavior
stay stable over time, then revisit structured expansion.
