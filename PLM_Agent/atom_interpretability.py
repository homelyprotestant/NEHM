"""Self-contained PLM_Agent global-dictionary interpretability helpers.

Implements the particle/fiber microscopy language-model pipeline:
  1. approximate K-SVD + non-negative Elastic Net sparse coding
  2. top-K atom retrieval over the corpus
  3. local Qwen RCS over Material_Database metadata (excluding logit fields)
  4. cloud GPT synthesizer for reusable per-atom labels
  5. sample description: FAISS L2 + vision lead type/appearance; atoms critical but interpreted
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .classical_ksvd import (
    ClassicalKsvdConfig,
    encode_samples,
    fit_approx_dictionary,
    history_to_dataframe,
    load_classical_ksvd_config,
    reconstruction_stats,
    save_classical_ksvd_config,
)
from .config import load_env
from .json_utils import extract_json_object
from .llm_clients import chat_text, chat_vision
from .text_chunking import SplitParagraph, atomize_with_overlap

LOGIT_FIELD_PATTERN = re.compile(r"(?i)^logit|logit$|_logit$|^run\d+_logit")

RCS_MODEL_DEFAULT = "qwen2.5:7b"
COORDINATOR_MODEL_DEFAULT = "gpt-5.1"
LOCAL_COORDINATOR_MODEL_DEFAULT = "qwen3.8:27b"
TOP_K_DEFAULT = 40
N_ATOMS_DEFAULT = 400

# Visual-History Run4 Elastic Net / approx K-SVD settings, with n_atoms=400.
NEHM_KSVD_CONFIG = ClassicalKsvdConfig(
    input_dim=2304,
    source_input_dim=2304,
    embedding_block="full",
    block_start=0,
    block_end=2304,
    n_atoms=N_ATOMS_DEFAULT,
    max_iter=10,
    ksvd_iter=1,
    sparse_method="elastic_net",
    omp_n_nonzero=40,
    lasso_alpha=0.5,
    lasso_algorithm="lasso_cd",
    elastic_net_alpha=1e-4,
    elastic_net_l1_ratio=0.5,
    elastic_net_max_iter=1000,
    sparse_codes_nonnegative=True,
    refine_r_iters=0,
    refine_r_lr=0.01,
    refine_r_lambda=0.5,
    preprocess_mode="raw",
    subsample_fraction=0.1,
    active_eps=1e-4,
    reinit_dead_atoms=True,
    min_dict_change=1e-5,
    patience=6,
    seed=42,
    prune_enabled=True,
    min_atom_support=20,
    max_top5_concentration=0.4,
    prune_every_iters=3,
    replacement_mode="residual_svd",
    residual_pool_size=200,
    max_prune_per_cycle=50,
    max_abs_code=1e4,
)

MICROSCOPY_RCS_QUESTION = """Which passages provide detailed, concrete visual evidence about
how the specimen looks under optical microscopy? Prioritize particle vs fiber morphology,
color, relief, birefringence / extinction behavior, pleochroism, crystal habit, grain size,
aggregation, surface texture, inclusions, and illumination-dependent appearance (PPL/XPL/RPL).
Catalogue chemistry, refractive index, and class labels are useful when they clarify those
visible qualities. Pure taxonomy without visible description is weak evidence."""

RCS_PROMPT = """You are the Qwen RCS passage analyst and an expert optical microscopist.
Assess one Material Database specimen-metadata chunk for its usefulness in interpreting a
learned visual dictionary atom from microscopy embeddings.

Return ONLY valid JSON:
{{
  "relevance_score": <integer 1-10>,
  "contextual_sentences": [
    "<one complete sentence preserving concrete microscopic / morphological evidence>",
    "<include 1-8 entries, matching the amount of evidence in the passage>"
  ],
  "material_terms": []
}}

Scoring:
- 8-10: specific evidence about morphology (particle/fiber/crystal habit), color, relief,
  birefringence, pleochroism, texture, aggregation, illumination response, or grain size.
- 4-7: useful specimen identity / class / composition context with limited visual detail.
- 1-3: primarily opaque labels, empty fields, or non-visual metadata.
- Use only the passage. Do not invent pigments, optics, or morphology.
- Prefer precise visible particulars over generic phrases.
- Return 1-8 `contextual_sentences`. Each entry must be exactly one complete sentence.

Research question:
{question}

Passage metadata:
Specimen: {specimen}
Source: {source}
Illumination: {illumination}
Image: {image}

Passage text:
{text}
"""

COORDINATOR_RERANK_PROMPT = """You are the cloud reranker for one learned visual dictionary
atom from an optical-microscopy embedding space. Its mean-centered 2304-D vector retrieved
40 Material Database specimens. Qwen extracted and scored microscopic evidence from their
metadata chunks. Candidates remain in cosine-retrieval order.

Task:
Rerank all 40 candidates by usefulness for interpreting the atom, jointly considering
atom association (cosine), concrete microscopic evidence, and cross-candidate recurrence.
Do not copy either cosine order or Qwen-score order mechanically.

Return ONLY valid JSON containing every CANDIDATE number exactly once:
{{
  "reranked_description_ranks": [<40 integer CANDIDATE numbers, most to least useful>]
}}

Cosine-ordered candidates with Qwen evidence:
{evidence}
"""

COORDINATOR_SYNTHESIS_PROMPT = """You are the cloud synthesizer labeling one learned visual
dictionary atom from optical microscopy of particles and fibers. The 40 candidates below
have already been reranked by the cloud coordinator from most to least useful.

Task:
Synthesize a precise microscopy-oriented label and a detailed 4-8 sentence description of
the recurring visual qualities supported across multiple specimens. Emphasize morphology
(particle vs fiber vs aggregate), color, relief, birefringence / extinction, pleochroism,
habit, texture, and illumination-dependent appearance.

Critical rules:
- Use ONLY the supplied Qwen summaries, which derive only from Material Database metadata.
- Treat cosine retrieval as atom association and Qwen RCS as usefulness; neither score is
  itself visual evidence.
- Aggregate across multiple specimens; never center the answer on one row.
- Do not invent optical constants, pigments, or morphology absent from the evidence.
- Prefer exact evidence over umbrella phrases such as "crystalline particles" unless you
  immediately state the concrete supporting features.
- Claim a recurring quality only when at least two candidates support it.
- If evidence is thin, state a cautious reading and set confidence to "low".

Return ONLY valid JSON:
{{
  "label": "<3-8 word microscopy-oriented label>",
  "microscopy_sentences": [
    "<one complete evidence-dense sentence>",
    "<continue until there are 4-8 array entries total>"
  ],
  "confidence": "high|medium|low",
  "evidence_summary": [
    "<feature; supported by CANDIDATE numbers X and Y>"
  ]
}}

Coordinator-reranked candidates with Qwen evidence:
{evidence}
"""

VISUAL_OBSERVATION_PROMPT = """Inspect only the supplied optical-microscopy image. Do not use
filename, directory, database labels, atoms, or nearest neighbors. Report conservative
positive visual observations (what is visible), not long lists of what the sample is not.

Style preference: favor affirmative descriptions and avoid long comparisons with absent
alternatives. Phrases such as "rather than", "non-fibrous", and "mineral shards" are
permitted when they make the microscopy reading clearer; they must not by themselves cause
a rejected draft. If a specimen feature is not visible, usually omit it. For
`artifacts_or_overlays`, return an empty list when none are visible; do not write an absence
statement such as "no visible scale bar."

Focus on:
- particle vs fiber vs mixed morphology
- color / opacity / relief appearance under the visible illumination
- grain size relative to the field
- estimated magnification (approximate objective / total range or low/medium/high power),
  inferred conservatively from field coverage, apparent particle/fiber scale, and detail
- aggregation, habit, edges, texture
- any clear birefringent / anisotropic cues if the background and highlights support them
- image artifacts (compression, glare, scale bar, figure number overlays)

Return JSON only:
{{
  "morphology": {{
    "primary": "particle|fiber|aggregate|mixed|uncertain",
    "confidence": "low|medium|high",
    "notes": ["direct visible cues"]
  }},
  "color_and_tone": ["conservative color / opacity / relief observations"],
  "size_and_distribution": ["relative grain size, density, clustering"],
  "magnification_estimate": {{
    "estimate": "approximate magnification or low|medium|high power",
    "confidence": "low|medium|high",
    "basis": "positive visual cues supporting the estimate"
  }},
  "optical_cues": ["possible birefringence / anisotropy / extinction cues if visible"],
  "artifacts_or_overlays": ["scale bars, figure numbers, compression, glare"],
  "positive_constraints": {{
    "supported": ["broad morphological / optical classes supported by the image"],
    "summary": "short positive visual reading"
  }},
  "uncertainties": ["genuine visual ambiguities"]
}}
"""

SAMPLE_RERANK_PROMPT = """You coordinate dictionary-atom evidence for one query specimen.
FAISS nearest neighbors and direct vision establish type and visible appearance; the atoms are
still critically important and must be interpreted carefully for synthesis.

The atoms below are active in the specimen's non-negative Elastic Net sparse code. No class
or specimen identity was used to choose them.

Rerank every atom for later synthesis. Prefer coefficient weight. Interpret atom labels with
care: an atom labeled "fiber" on a particle specimen (or vice versa), or a material-specific
label that disagrees with L2/vision identity, may still encode shared color, interference
contrast, illumination response, relief, habit, texture, or surface appearance (e.g. pitting
from delustrant vs. "pitted softwood tracheid" wording). Do not discard such atoms; rank them
for those transferable attributes. Do not promote an atom solely because its type or material
headline matches or mismatches the specimen.

Return JSON only:
{{
  "reranked_atom_ids": [every supplied integer atom ID exactly once],
  "rationale": ["brief explanation of the weighted / interpretive ordering"]
}}

TARGET IDENTIFIER
{target}

ESTABLISHED TYPE HINT FROM L2 + VISION (may be empty)
{type_hint}

ORIGINAL WEIGHTED GLOBAL ATOMS
{atoms}
"""

SAMPLE_SYNTHESIS_PROMPT = """Write a grounded optical-microscopy description of the query
specimen. Use three evidence streams together, with different roles:

Opening requirements (hard constraints):
- Name the target's recorded illumination modality explicitly in sentence 1.
- State an estimated magnification in sentence 1 or 2. Use the recorded magnification as
  the primary anchor when supplied; otherwise use the direct-vision estimate. Phrase it as
  an estimate (for example, "at approximately 40×" or "at medium magnification").
- These opening details must describe how the modality and scale shape the visible reading.

1. FAISS nearest neighbors (lead for type / identity context): use recurring specimen
   identity, class/subclass, composition, illumination, and catalog morphology across
   neighbors to establish what the sample most likely is. Give greatest identity weight to
   the specimen repeated most often among the nearest neighbors; use distance to break count
   ties. Each neighbor includes a `chemical_formula` value from the database.
   The reference corpus is not a prevalence sample: repeated database coverage does not make
   a material common in artworks. Never promote a material identity from row count or shared
   color alone.
   When `strong_identity_family_consensus` is present, explicitly state that identity family
   as the leading interpretation. Treat lazurite as the characteristic blue mineral phase of
   ultramarine, so agreement across those names supports "ultramarine/lazurite" language.
   If `repeated_formula_suggestion` is present in the supplied L2 consensus, explicitly
   suggest that formula in the description as a likely composition and cite it [FAISS].
   Do not invent or suggest a formula that fails the repeat threshold.
   If `identity_inference_allowed` is false, the neighbors are visual analogs only:
   describe shared appearance, but do not assign any neighbor material identity or chemical
   formula to the uploaded sample.
   In reader-facing prose, always call them "the nearest neighbors"; never write
   "the L2 neighborhood."
2. Direct vision (lead for what is visible in the query image): confirm / refine particle
   vs fiber (or mixed), color, relief, habit, aggregation, and optical cues.
3. Dictionary atoms (critical interpretive evidence): the weighted Elastic Net atoms are a
   core account of how this embedding decomposes. You MUST interpret them — do not ignore
   high-weight atoms and do not copy their labels verbatim as specimen identity. Read each
   atom for transferable microscopy attributes (color, interference / birefringence cues,
   illumination response, relief, texture, surface pitting / mottling, habit particulars,
   aggregation). If an atom's headline type or material name disagrees with the type
   established from L2 + vision, keep the established identity and still mine the atom for
   compatible visual attributes.
   Example: an atom labeled "Pitted softwood tracheid pulp fibers" on an acrylic fiber may
   be pointing to a pitted / mottled surface appearance (e.g. TiO2 delustrant), not claiming
   the specimen is softwood pulp. Cite the atom for that interpreted attribute on THIS
   sample. NEVER write a contrast against softwood/plant fibers ("contrasts with tracheids",
   "bordered pits not evident here") — that is forbidden negative evidence.
   Cite atoms when their interpreted attributes enter the description.

Positive-evidence rule (strong style preference, with hard validation limited to substantive
denial catalogs):
- State ONLY what the sample IS and DOES show. Never define it by denying another class,
  listing absent particle types, or "distinguishing" this mount from other preparations.
- Forbidden contrastive / negative / foil phrasing (non-exhaustive) — rewrite positively:
  sentence-initial "No/None …", "not observed/seen/present/evident/visible",
  "as opposed to", "instead of", "unlike", "not like", "contrasts with",
  "in contrast", "compared to/with", "whereas", "while other", "distinguishing … from",
  "distinguishes this", "is not", "are not", "not a/an", "without", "free of",
  "inclusion-free", "no evidence", "ruled out", "excludes", "absence of", "lacks",
  "does not appear", "far from", or long catalogs of absent materials.
  Bad: "No discrete angular mineral shards … gypsum grains are observed … distinguishing
  this preparation from mounts where … particulate phases adjacent to fibers."
  Good: "The field shows continuous fiber filaments under the recorded illumination, with
  surface relief and interference colors consistent with the acrylic neighbors. [FAISS] [Vision]"
  Bad: "The smooth perimeter contrasts with pitted softwood tracheid pulps … not evident here."
  Good: "The acrylic fiber has a smooth perimeter and a finely pitted / mottled surface
  consistent with delustrant. [Vision] [Atoms: N]"
  Bad: "elongate cellulose fibers rather than strongly colored mineral shards"
  Good: "elongate cellulose fibers with pale interference colors under polarized light"
- Do not name alternate materials, habits, or atom source-classes as foils. If a mismatched
  atom contributes, name the attribute on this sample only.
- Prefer affirmative microscopy language: morphology, color, relief, optics, habit,
  texture, aggregation, illumination response.

Citation tags — end every sentence with source tags immediately before final punctuation:
  [FAISS], [Vision], and/or [Atoms: 11, 16].
- Lead the account with [FAISS] and [Vision] for type and direct appearance.
- Atom citations are required whenever an interpreted atom attribute is used; a substantial
  share of sentences should include [Atoms: …] when high-weight atoms contribute.
- Cite each atom ID at most ONCE in the entire `microscopy_description`. Combine all
  attributes supported by one atom into its single cited sentence. After atom 92 appears
  in one `[Atoms: 92, ...]` tag, atom 92 must never appear in another tag or sentence.
- Atoms contributing at least 1% are candidate evidence, not a mandatory citation list.
  Select only atoms that add compatible, useful attributes to this sample. Every atom that
  is actually cited must appear exactly once in `microscopy_description` and must have one
  matching `atom_contributions` entry with an affirmative transferable attribute. Never
  explain a rejected atom identity or type;
- Include at least 10 distinct entries in `atom_contributions`, selected from the supplied
  ranked atoms. These are the atoms explicitly considered in the interpretation panel.
  They do not all need citations in `microscopy_description`; cite an atom only when its
  interpreted attribute is actually used in the prose.
- Exception: the TOP-RANKED atom is mandatory. Use its exact `label` phrase in the natural
  language of `microscopy_description`, interpret that phrase as a feature of this sample,
  and cite its atom ID exactly once. For example, a top label of "Starch granules with
  Maltese crosses" must contribute that wording and its optical/morphological meaning.
  never use phrases such as "identity is not imposed", "type conflicts", "suppressed",
  or "deferred." Translate the atom into the compatible feature it contributes
  feature it contributes (e.g. "supports blue particle color" or "supports surface pitting").

Return JSON only:
{{
  "microscopy_description": "6-10 coherent sentences",
  "established_type": "particle|fiber|aggregate|mixed|uncertain",
  "suggested_chemical_formula": "repeated L2 formula or empty string",
  "magnification_estimate": "concise estimated magnification and basis",
  "formal_elements": {{
    "morphology": "concise positive account",
    "color_relief_optics": "concise positive account",
    "habit_texture_aggregation": "concise positive account",
    "illumination_response": "concise positive account"
  }},
  "uncertainties": ["specific limitations; keep short; avoid long negative catalogs"],
  "atom_contributions": [
    {{"atom_id": 0, "importance_percent": 0.0, "contribution": "one affirmative transferable attribute contributed to this sample"}}
  ]
}}

TARGET IDENTIFIER
{target}

FAISS NEAREST NEIGHBORS (TYPE / IDENTITY LEAD)
{neighbors}

L2 FREQUENCY CONSENSUS (COUNT FIRST, DISTANCE TIE-BREAK)
{l2_consensus}

DIRECT VISUAL OBSERVATIONS (IMAGE LEAD)
{visual_observations}

DICTIONARY ATOMS — CRITICAL, INTERPRET CAREFULLY
{atoms}

TOP-RANKED ATOM — TITLE MUST APPEAR IN THE DESCRIPTION AND ID MUST BE CITED ONCE
{top_atom}

ATOMS AT OR ABOVE 1% CONTRIBUTION — CANDIDATES; CITE ONLY RELEVANT CONTRIBUTORS
{required_high_weight_atoms}
"""

# Contrastive / negative foils that must not appear in sample synthesis prose.
_NEGATIVE_CONTRAST_RE = re.compile(
    r"(?ix)"
    r"("
    # Sentence- or clause-initial denial catalogs
    r"(?:^|[\n.:;!?]\s*)(?:no|none)\s+"
    r"|"
    r"\bas\ opposed\ to\b|\binstead\ of\b|\bunlike\b|\bnot\ like\b"
    r"|"
    r"\bcontrasts\ with\b|\bin\ contrast\b|\bcompared\ to\b|\bcompared\ with\b"
    r"|"
    r"\bwhereas\b|\bwhile\ other\b|\bas\ opposed\b"
    r"|"
    r"\bdistinguish(?:es|ing|ed)?\b.{0,80}\bfrom\b"
    r"|"
    r"\bis\ not\b|\bare\ not\b|\bwas\ not\b|\bwere\ not\b|\bisn't\b|\baren't\b"
    r"|"
    r"\bnot\ a\b|\bnot\ an\b|\bnot\ evident\b|\bnot\ visible\b|\bnot\ present\b"
    r"|"
    r"\bnot\ observed\b|\bnot\ seen\b|\bare\ observed\b.{0,40}\bdistinguishing\b"
    r"|"
    r"\bno\ clear\b"
    r"|"
    r"\bwithout\ being\b|\bwithout\b|\bfree\ of\b|\binclusion-free\b"
    r"|"
    r"\bno\ evidence\b|\bruled\ out\b|\bexcludes\b|\babsence\ of\b|\blacks\b"
    r"|"
    r"\bdoes\ not\ appear\b|\bdo\ not\ appear\b|\bfar\ from\b|\bnone\ of\ the\b"
    r"|"
    r"\bneither\b.+\bnor\b|\babsent\ here\b|\bmissing\ here\b|\babsent\b"
    r"|"
    r"\bother\ structured\b|\bother\ plant\b"
    r"|"
    # Denial catalogs of alternate particle types (ordinary references to
    # mineral shards are permitted and never fail validation on their own).
    r"\b(?:no|none|not)\b.{0,100}\b(?:fragments?|gypsum|carbonate|particulate\ phases)\b"
    r")"
)


def _collect_all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(_collect_all_strings(child))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for child in value:
            out.extend(_collect_all_strings(child))
        return out
    return []


def _validate_positive_text(text: str, *, stage: str) -> None:
    hit = _NEGATIVE_CONTRAST_RE.search(text)
    if hit:
        snippet = text[max(0, hit.start() - 40) : hit.end() + 60].replace("\n", " ")
        raise ValueError(
            f"{stage} positive-evidence violation near {hit.group(0)!r}: …{snippet}…"
        )


def _validate_positive_sample_prose(payload: dict[str, Any]) -> None:
    # Auxiliary bookkeeping should never make an otherwise valid description fail.
    # Replace negative atom-accounting prose with a neutral affirmative record, and omit
    # negative uncertainty clauses. The reader-facing description remains strictly checked.
    contributions = payload.get("atom_contributions")
    if isinstance(contributions, list):
        for item in contributions:
            if not isinstance(item, dict):
                continue
            text = str(item.get("contribution", ""))
            if _NEGATIVE_CONTRAST_RE.search(text):
                item["contribution"] = (
                    "Contributes a compatible visual attribute from the weighted atom "
                    "evidence to the sample interpretation."
                )

    uncertainties = payload.get("uncertainties")
    if isinstance(uncertainties, list):
        payload["uncertainties"] = [
            item
            for item in uncertainties
            if not _NEGATIVE_CONTRAST_RE.search(str(item))
        ]

    reader_prose = [str(payload.get("microscopy_description", ""))]
    formal = payload.get("formal_elements") or {}
    if isinstance(formal, dict):
        reader_prose.extend(str(value) for value in formal.values())
    _validate_positive_text("\n".join(reader_prose), stage="sample synthesis")
    _validate_unique_atom_citations(str(payload.get("microscopy_description", "")))


_ATOM_CITATION_RE = re.compile(r"\[Atoms:\s*([0-9,\s]+)\]", re.IGNORECASE)


def _validate_unique_atom_citations(description: str) -> None:
    seen: set[int] = set()
    repeated: set[int] = set()
    for match in _ATOM_CITATION_RE.finditer(description):
        for raw in match.group(1).split(","):
            raw = raw.strip()
            if not raw:
                continue
            atom_id = int(raw)
            if atom_id in seen:
                repeated.add(atom_id)
            seen.add(atom_id)
    if repeated:
        ids = ", ".join(str(atom_id) for atom_id in sorted(repeated))
        raise ValueError(
            "duplicate atom citation violation: each atom may be cited once; "
            f"repeated IDs: {ids}"
        )


def _validate_top_atom_use(description: str, top_atom: dict[str, Any]) -> None:
    atom_id = int(top_atom["atom_id"])
    label = safe_text(top_atom.get("label", "")).strip()
    cited_ids: list[int] = []
    for match in _ATOM_CITATION_RE.finditer(description):
        cited_ids.extend(
            int(raw.strip())
            for raw in match.group(1).split(",")
            if raw.strip()
        )
    if cited_ids.count(atom_id) != 1:
        raise ValueError(
            "top atom violation: top-ranked atom "
            f"{atom_id} must be cited exactly once"
        )
    normalized_description = re.sub(r"\s+", " ", description).casefold()
    normalized_label = re.sub(r"\s+", " ", label).casefold()
    if label and normalized_label not in normalized_description:
        raise ValueError(
            "top atom violation: microscopy_description must use the exact top atom "
            f"label phrase {label!r}"
        )


def _validate_neighbor_wording(description: str) -> None:
    if re.search(r"\b(?:the\s+)?L2\s+neighbou?rhood\b", description, re.IGNORECASE):
        raise ValueError(
            'neighbor wording violation: use "the nearest neighbors" instead of '
            '"the L2 neighborhood"'
        )


def _validate_opening_context(
    description: str,
    *,
    illumination: str,
    magnification: str,
) -> None:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", description)
        if sentence.strip()
    ]
    opening = " ".join(sentences[:2]).lower().replace("×", "x")
    illumination_text = safe_text(illumination).lower()
    illumination_tokens = re.findall(r"[a-z0-9]+", illumination_text)
    if illumination_tokens and illumination_text not in {"unspecified", "unknown"}:
        if not all(token in opening for token in illumination_tokens):
            raise ValueError(
                "opening context violation: sentence 1 must name illumination modality "
                f"{illumination!r}"
            )

    magnification_text = safe_text(magnification).lower().replace("×", "x").replace(" ", "")
    if magnification_text and magnification_text not in {"unspecified", "unknown"}:
        numeric = re.findall(r"\d+(?:\.\d+)?", magnification_text)
        opening_compact = opening.replace(" ", "")
        magnitude_present = (
            any(f"{number}x" in opening_compact for number in numeric)
            if numeric
            else magnification_text in opening_compact
        )
        if not magnitude_present:
            raise ValueError(
                "opening context violation: sentence 1 or 2 must state estimated "
                f"magnification near {magnification!r}"
            )
    elif not re.search(
        r"\b(?:magnification|low[- ]power|medium[- ]power|high[- ]power|"
        r"approximately\s+\d+\s*x|\d+\s*x)\b",
        opening,
    ):
        raise ValueError(
            "opening context violation: sentence 1 or 2 must include a magnification estimate"
        )


def _validate_repeated_formula_suggestion(
    payload: dict[str, Any],
    repeated_formula: dict[str, Any] | None,
) -> None:
    if repeated_formula is None:
        return
    formula = safe_text(repeated_formula.get("value", "")).strip()
    description = safe_text(payload.get("microscopy_description", ""))
    normalized_formula = re.sub(r"\s+", "", formula).casefold()
    normalized_description = re.sub(r"\s+", "", description).casefold()
    if normalized_formula not in normalized_description:
        raise ValueError(
            "L2 consensus violation: the repeatedly occurring chemical formula "
            f"{formula!r} must be suggested in microscopy_description"
        )
    suggested = safe_text(payload.get("suggested_chemical_formula", "")).strip()
    if re.sub(r"\s+", "", suggested).casefold() != normalized_formula:
        raise ValueError(
            "L2 consensus violation: suggested_chemical_formula must equal "
            f"{formula!r}"
        )


def _validate_retrieval_confidence(
    payload: dict[str, Any],
    l2_consensus: dict[str, Any],
) -> None:
    if bool(l2_consensus.get("identity_inference_allowed", True)):
        return
    suggested = safe_text(payload.get("suggested_chemical_formula", "")).strip()
    if suggested:
        raise ValueError(
            "retrieval confidence violation: low-confidence nearest neighbors cannot "
            "supply a chemical formula"
        )
    description = safe_text(payload.get("microscopy_description", "")).casefold()
    for formula in l2_consensus.get("formula_frequencies", []):
        value = safe_text(formula.get("value", "")).strip()
        if value and re.sub(r"\s+", "", value).casefold() in re.sub(
            r"\s+", "", description
        ):
            raise ValueError(
                "retrieval confidence violation: low-confidence nearest neighbors cannot "
                f"assign formula {value!r}"
            )


def _validate_identity_family_consensus(
    payload: dict[str, Any],
    l2_consensus: dict[str, Any],
) -> None:
    family = l2_consensus.get("strong_identity_family_consensus")
    if not isinstance(family, dict):
        return
    family_name = safe_text(family.get("value", "")).strip()
    description = safe_text(payload.get("microscopy_description", "")).casefold()
    required_names = [
        part.strip().casefold() for part in family_name.split("/") if part.strip()
    ]
    if not required_names or not all(name in description for name in required_names):
        raise ValueError(
            "identity family violation: microscopy_description must explicitly state "
            f"the strong nearest-neighbor consensus {family_name!r}"
        )


def _validate_minimum_atom_contributions(
    payload: dict[str, Any],
    ranked_atoms: list[dict[str, Any]],
    *,
    minimum: int = 10,
) -> None:
    contributions = payload.get("atom_contributions")
    if not isinstance(contributions, list):
        raise ValueError("atom consideration violation: atom_contributions must be a list")
    required_count = min(int(minimum), len(ranked_atoms))
    candidate_ids = {int(atom["atom_id"]) for atom in ranked_atoms}
    contribution_ids = [
        int(item["atom_id"])
        for item in contributions
        if isinstance(item, dict) and "atom_id" in item
    ]
    valid_unique_ids = set(contribution_ids) & candidate_ids
    if len(valid_unique_ids) < required_count:
        raise ValueError(
            "atom consideration violation: atom_contributions must contain at least "
            f"{required_count} distinct supplied atoms"
        )


def _strip_negative_strings(value: Any) -> Any:
    """Recursively omit negative/contrastive prose from vision JSON."""
    if isinstance(value, str):
        return None if _NEGATIVE_CONTRAST_RE.search(value) else value
    if isinstance(value, list):
        cleaned = [_strip_negative_strings(child) for child in value]
        return [child for child in cleaned if child is not None]
    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, child in value.items():
            cleaned = _strip_negative_strings(child)
            if cleaned is not None:
                cleaned_dict[str(key)] = cleaned
        return cleaned_dict
    return value


def _validate_positive_visual_prose(payload: dict[str, Any]) -> None:
    # Vision models sometimes put an absence statement in an unexpected field. Remove all
    # negative/contrastive strings recursively rather than spending retries on harmless
    # overlay notes or allowing them to contaminate the final synthesis prompt.
    cleaned = _strip_negative_strings(payload)
    if not isinstance(cleaned, dict):
        raise ValueError("Direct-vision payload could not be sanitized")
    payload.clear()
    payload.update(cleaned)

    # Sanity-check the cleaned payload; this should not trigger ordinary retries.
    _validate_positive_text(
        "\n".join(_collect_all_strings(payload)),
        stage="direct vision",
    )


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def is_logit_field(name: str) -> bool:
    return bool(LOGIT_FIELD_PATTERN.search(str(name).strip()))


def metadata_fields(columns: Iterable[str]) -> list[str]:
    return [str(c) for c in columns if not is_logit_field(str(c))]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).replace("\x00", " ").strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def row_metadata_blob(row: pd.Series, fields: list[str]) -> str:
    parts: list[str] = []
    for col in fields:
        val = safe_text(row.get(col, ""))
        if not val:
            continue
        parts.append(f"{col}: {val}")
    return "\n".join(parts)


def description_chunks(text: str) -> list[str]:
    """Chunk metadata text into small overlapping windows."""
    text = text.strip()
    if not text:
        return []
    try:
        source = SplitParagraph(page=1, text=text, local_index=1, section="description")
        chunks = atomize_with_overlap([source])
        out = [chunk.text.strip() for chunk in chunks if chunk.text.strip()]
        return out or [text]
    except Exception:
        # Fallback: paragraph / hard wrap
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(paras) <= 1 and len(text) > 1200:
            return [text[i : i + 900] for i in range(0, len(text), 800)]
        return paras or [text]


def cache_paths(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "out_dir": out_dir,
        "config": out_dir / "global_ksvd_config.json",
        "dictionary": out_dir / "global_dictionary_atoms.npy",
        "mean": out_dir / "global_embedding_mean.npy",
        "codes": out_dir / "global_sparse_codes.npy",
        "history": out_dir / "global_ksvd_history.csv",
        "meta": out_dir / "global_dictionary_meta.json",
        "usage": out_dir / "global_atom_usage.csv",
        "retrieval": out_dir / "atom_top40_retrieval.jsonl",
        "rcs_cache": out_dir / "qwen_rcs_chunk_cache.jsonl",
        "atom_labels": out_dir / "atom_microscopy_labels.json",
        "atom_labels_jsonl": out_dir / "atom_microscopy_labels.jsonl",
        "sample_dir": out_dir / "sample_descriptions",
    }


def load_aligned_corpus(
    *,
    z_path: Path,
    xlsx_path: Path,
    sheet_name: str = "Sheet2",
    images_dir: Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    z = np.asarray(np.load(z_path), dtype=np.float32)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name).reset_index(drop=True)
    n = min(len(df), int(z.shape[0]))
    if len(df) != int(z.shape[0]):
        progress(
            f"Aligning corpus to min(db={len(df)}, z={z.shape[0]}) = {n} rows"
        )
    df = df.iloc[:n].copy()
    z = z[:n]
    fields = metadata_fields(df.columns)
    if images_dir is not None and "Image" in df.columns:
        missing = 0
        for name in df["Image"].fillna("").astype(str):
            if name and not (images_dir / name).is_file():
                missing += 1
        progress(f"Image files missing under {images_dir}: {missing}/{n}")
    return z, df, fields


def train_or_load_dictionary(
    z: np.ndarray,
    out_dir: Path,
    *,
    cfg: ClassicalKsvdConfig | None = None,
    force_retrain: bool = False,
) -> dict[str, Any]:
    paths = cache_paths(out_dir)
    cfg = cfg or NEHM_KSVD_CONFIG
    have_cache = (
        paths["dictionary"].is_file()
        and paths["mean"].is_file()
        and paths["codes"].is_file()
        and paths["config"].is_file()
        and not force_retrain
    )
    if have_cache:
        progress(f"Loading cached dictionary from {paths['out_dir']}")
        d = np.load(paths["dictionary"]).astype(np.float32)
        mean = np.load(paths["mean"]).astype(np.float32)
        codes = np.load(paths["codes"]).astype(np.float32)
        cfg = load_classical_ksvd_config(paths["config"])
        return {"D": d, "mean": mean, "codes": codes, "cfg": cfg, "paths": paths, "cached": True}

    progress(
        f"Training approx K-SVD | n_atoms={cfg.n_atoms} samples={z.shape[0]} "
        f"elastic_net α={cfg.elastic_net_alpha} l1_ratio={cfg.elastic_net_l1_ratio} "
        f"nonnegative={cfg.sparse_codes_nonnegative}"
    )
    mean = np.mean(z, axis=0).astype(np.float32)
    x = (z - mean[None, :]).astype(np.float32)
    d, history = fit_approx_dictionary(
        x,
        cfg,
        verbose=True,
        checkpoint_dir=paths["out_dir"],
        checkpoint_every_iters=1,
    )
    progress("Encoding full corpus with non-negative Elastic Net")
    codes = encode_samples(x, d, cfg, show_progress=True)
    stats = reconstruction_stats(x, d, codes, active_eps=float(cfg.active_eps), cfg=cfg)

    np.save(paths["dictionary"], d.astype(np.float32))
    np.save(paths["mean"], mean.astype(np.float32))
    np.save(paths["codes"], codes.astype(np.float32))
    save_classical_ksvd_config(cfg, paths["config"])
    history_to_dataframe(history).to_csv(paths["history"], index=False)
    usage = (
        pd.DataFrame(
            {
                "atom": np.arange(codes.shape[1]),
                "usage_count": (codes > float(cfg.active_eps)).sum(axis=0),
                "mean_code": codes.mean(axis=0),
                "max_code": codes.max(axis=0),
            }
        )
        .sort_values("usage_count", ascending=False)
    )
    usage.to_csv(paths["usage"], index=False)
    meta = {
        "n_atoms": int(cfg.n_atoms),
        "n_samples": int(z.shape[0]),
        "embedding_dim": int(z.shape[1]),
        "mean_centered": True,
        "recon_mse": float(stats["mse"]),
        "mean_active_atoms_per_sample": float(stats["mean_active"]),
        "config": asdict(cfg),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    progress(
        f"Saved dictionary | mse={stats['mse']:.6f} "
        f"mean_active={stats['mean_active']:.1f} dead={stats['dead_atoms']}"
    )
    return {"D": d, "mean": mean, "codes": codes, "cfg": cfg, "paths": paths, "cached": False, "meta": meta}


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x64, axis=1)
    norms = np.maximum(norms, 1e-12)
    return x64 / norms[:, None]


def retrieve_top_hits_for_atoms(
    dictionary: np.ndarray,
    z: np.ndarray,
    mean: np.ndarray,
    df: pd.DataFrame,
    fields: list[str],
    *,
    top_k: int = TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    centered = np.asarray(z, dtype=np.float64) - np.asarray(mean, dtype=np.float64)[None, :]
    finite = np.isfinite(centered).all(axis=1)
    rows = np.flatnonzero(finite)
    matrix_unit = _unit_rows(centered[rows])
    dict_unit = _unit_rows(dictionary)
    sims = np.einsum("ij,kj->ik", dict_unit, matrix_unit, optimize=False)
    top_pos = np.argsort(sims, axis=1)[:, ::-1][:, : int(top_k)]

    retrieval: list[dict[str, Any]] = []
    for atom_id in range(dictionary.shape[0]):
        hits: list[dict[str, Any]] = []
        for rank, pos in enumerate(top_pos[atom_id], start=1):
            db_row = int(rows[int(pos)])
            row = df.iloc[db_row]
            blob = row_metadata_blob(row, fields)
            hits.append(
                {
                    "retrieval_rank": rank,
                    "database_row": db_row,
                    "cosine": float(sims[atom_id, int(pos)]),
                    "specimen": safe_text(row.get("Specimen Name", "")),
                    "source": safe_text(row.get("Source", "")),
                    "illumination": safe_text(row.get("Illumination Modality", "")),
                    "image": safe_text(row.get("Image", "")),
                    "metadata_blob": blob,
                    "chunks": description_chunks(blob),
                }
            )
        retrieval.append({"atom_id": int(atom_id), "hits": hits})
    return retrieval


def save_retrieval_jsonl(path: Path, retrieval: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for atom in retrieval:
            # Keep chunks for RCS, but avoid huge redundant blobs in one line if empty.
            f.write(json.dumps(atom, ensure_ascii=False) + "\n")


def load_retrieval_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def chunk_key(text: str, *, version: str = "nehm-materials-rcs-v1") -> str:
    digest = hashlib.sha1(f"{version}\n{text.strip()}".encode("utf-8")).hexdigest()
    return digest


def load_jsonl_map(path: Path, key_field: str = "chunk_key") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[str(rec[key_field])] = rec
    return out


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_qwen_rcs_for_retrieval(
    retrieval: list[dict[str, Any]],
    rcs_cache_path: Path,
    *,
    question: str = MICROSCOPY_RCS_QUESTION,
    max_atoms: int | None = None,
) -> dict[str, dict[str, Any]]:
    load_env()

    cache = load_jsonl_map(rcs_cache_path)
    atoms = retrieval if max_atoms is None else retrieval[: int(max_atoms)]
    pending: list[tuple[str, dict[str, Any], str]] = []
    seen: set[str] = set()
    for atom in atoms:
        for hit in atom["hits"]:
            for chunk in hit.get("chunks", []):
                key = chunk_key(chunk)
                if key in cache or key in seen:
                    continue
                seen.add(key)
                pending.append((key, hit, chunk))

    progress(f"Qwen RCS | cached={len(cache)} pending={len(pending)}")
    for i, (key, hit, chunk) in enumerate(pending, start=1):
        prompt = RCS_PROMPT.format(
            question=question,
            specimen=hit.get("specimen", ""),
            source=hit.get("source", ""),
            illumination=hit.get("illumination", ""),
            image=hit.get("image", ""),
            text=chunk,
        )
        record: dict[str, Any] | None = None
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                raw = chat_text(
                    prompt,
                    backend="ollama",
                    model=RCS_MODEL_DEFAULT,
                    max_tokens=1024,
                    temperature=0.1,
                )
                payload = extract_json_object(raw)
                sentences = payload.get("contextual_sentences", [])
                if not isinstance(sentences, list):
                    raise ValueError("contextual_sentences must be a list")
                sentences = [str(s).strip() for s in sentences if str(s).strip()]
                if not (1 <= len(sentences) <= 8):
                    raise ValueError(f"expected 1-8 sentences, got {len(sentences)}")
                score = int(payload.get("relevance_score"))
                if not (1 <= score <= 10):
                    raise ValueError(f"invalid score {score}")
                record = {
                    "chunk_key": key,
                    "relevance_score": score,
                    "contextual_sentences": sentences,
                    "material_terms": payload.get("material_terms", []),
                    "specimen": hit.get("specimen", ""),
                    "source": hit.get("source", ""),
                }
                break
            except Exception as exc:
                last_err = exc
                time.sleep(min(8.0, 1.5 * (2**attempt)))
        if record is None:
            progress(f"RCS fallback after failures ({last_err})")
            record = {
                "chunk_key": key,
                "relevance_score": 1,
                "contextual_sentences": ["Little concrete microscopic evidence in this passage."],
                "material_terms": [],
                "specimen": hit.get("specimen", ""),
                "source": hit.get("source", ""),
                "fallback": True,
            }
        append_jsonl(rcs_cache_path, record)
        cache[key] = record
        if i % 25 == 0 or i == len(pending):
            progress(f"Qwen RCS progress {i}/{len(pending)}")
    return cache


def _enrich_hits(hits: list[dict[str, Any]], rcs_cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for hit in hits:
        summaries: list[str] = []
        scores: list[float] = []
        for chunk in hit.get("chunks", []):
            rec = rcs_cache.get(chunk_key(chunk))
            if not rec:
                continue
            scores.append(float(rec["relevance_score"]))
            summaries.extend(rec.get("contextual_sentences", []))
        best = max(scores) if scores else 1.0
        enriched.append(
            {
                **hit,
                "qwen_rcs_score": best,
                "qwen_rcs_summaries": summaries
                or ["No concrete microscopic evidence extracted."],
            }
        )
    return enriched


def _evidence_text(enriched_hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for hit in enriched_hits:
        summaries = hit["qwen_rcs_summaries"]
        summary_txt = " ".join(summaries[:6])
        blocks.append(
            f"CANDIDATE {hit['retrieval_rank']} | {hit.get('specimen','')} | "
            f"{hit.get('source','')} | {hit.get('illumination','')}\n"
            f"Qwen RCS {hit['qwen_rcs_score']:.1f}/10 | cosine {hit['cosine']:.6f}\n"
            f"{summary_txt}"
        )
    return "\n\n".join(blocks)


def _answer_chat(
    prompt: str,
    *,
    backend: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Route coordinator text to direct OpenAI or local Ollama."""
    return chat_text(
        prompt,
        backend=backend,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def synthesize_atom_labels(
    retrieval: list[dict[str, Any]],
    rcs_cache: dict[str, dict[str, Any]],
    labels_path: Path,
    labels_jsonl_path: Path,
    *,
    coordinator_backend: str = "openai",
    coordinator_model: str = COORDINATOR_MODEL_DEFAULT,
    max_atoms: int | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    load_env()
    existing: dict[int, dict[str, Any]] = {}
    if labels_jsonl_path.is_file() and not force:
        with labels_jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Atom labels define one stable vocabulary for the learned dictionary.
                # Reuse them regardless of the coordinator selected for sample synthesis.
                existing[int(rec["atom_id"])] = rec

    atoms = retrieval if max_atoms is None else retrieval[: int(max_atoms)]
    n_skip = sum(1 for a in atoms if int(a["atom_id"]) in existing and not force)
    n_todo = len(atoms) - n_skip
    progress(
        f"Atom labels | stable cache={len(existing)} generation_model={coordinator_model} "
        f"skip={n_skip} todo={n_todo} → {labels_jsonl_path}"
    )
    results: list[dict[str, Any]] = []
    done_new = 0
    for atom in atoms:
        atom_id = int(atom["atom_id"])
        if atom_id in existing and not force:
            results.append(existing[atom_id])
            continue
        done_new += 1
        progress(
            f"Labeling atom {atom_id} ({done_new}/{n_todo}) — "
            f"{coordinator_model} rerank…"
        )
        enriched = _enrich_hits(atom["hits"], rcs_cache)
        evidence = _evidence_text(enriched)

        # Rerank
        rerank_raw = _answer_chat(
            COORDINATOR_RERANK_PROMPT.format(evidence=evidence),
            backend=coordinator_backend,
            model=coordinator_model,
            max_tokens=4096,
            temperature=0.1,
        )
        rerank = extract_json_object(rerank_raw)
        ranks = [int(x) for x in rerank.get("reranked_description_ranks", [])]
        by_rank = {int(h["retrieval_rank"]): h for h in enriched}
        if sorted(ranks) != sorted(by_rank):
            # Fall back to original cosine order if malformed.
            ordered = enriched
            progress(f"Atom {atom_id}: rerank malformed — using cosine order")
        else:
            ordered = [by_rank[r] for r in ranks]
        ordered_evidence = _evidence_text(ordered)

        progress(
            f"Labeling atom {atom_id} ({done_new}/{n_todo}) — "
            f"{coordinator_model} synthesize…"
        )
        synth_raw = _answer_chat(
            COORDINATOR_SYNTHESIS_PROMPT.format(evidence=ordered_evidence),
            backend=coordinator_backend,
            model=coordinator_model,
            max_tokens=4096,
            temperature=0.1,
        )
        synth = extract_json_object(synth_raw)
        sentences = [str(s).strip() for s in synth.get("microscopy_sentences", []) if str(s).strip()]
        record = {
            "atom_id": atom_id,
            "rcs_model": RCS_MODEL_DEFAULT,
            "coordinator_backend": coordinator_backend,
            "coordinator_model": coordinator_model,
            "reranked_description_ranks": ranks,
            "label": str(synth.get("label", "")).strip(),
            "microscopy_description": " ".join(sentences),
            "microscopy_sentences": sentences,
            "confidence": str(synth.get("confidence", "low")).strip().lower(),
            "evidence_summary": synth.get("evidence_summary", []),
            "top_40_retrieval": [
                {
                    "retrieval_rank": h["retrieval_rank"],
                    "database_row": h["database_row"],
                    "cosine": h["cosine"],
                    "specimen": h.get("specimen", ""),
                    "source": h.get("source", ""),
                    "illumination": h.get("illumination", ""),
                    "qwen_rcs_score": h.get("qwen_rcs_score", 1.0),
                    "qwen_rcs_summaries": h.get("qwen_rcs_summaries", []),
                }
                for h in ordered
            ],
        }
        append_jsonl(labels_jsonl_path, record)
        existing[atom_id] = record
        results.append(record)
        progress(f"Labeled atom {atom_id}: {record['label']!r} ({record['confidence']})")

    progress(f"Wrote {len(results)} labels → {labels_path}")
    labels_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def load_atom_labels(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(rec["atom_id"]): rec for rec in data}


def active_weighted_atoms(
    code_row: np.ndarray,
    labels: dict[int, dict[str, Any]],
    *,
    active_eps: float = 1e-4,
    top_n: int | None = 25,
) -> list[dict[str, Any]]:
    code = np.asarray(code_row, dtype=np.float32)
    idxs = np.flatnonzero(code > float(active_eps))
    if idxs.size == 0:
        idxs = np.argsort(code)[::-1][:10]
    # Percentages always refer to the complete active sparse-code mass, even when the
    # returned list is truncated for display or prompting.
    total = float(code[idxs].sum()) if idxs.size else 1.0
    total = max(total, 1e-12)
    order = idxs[np.argsort(code[idxs])[::-1]]
    if top_n is not None:
        order = order[: int(top_n)]
    out: list[dict[str, Any]] = []
    for atom_id in order:
        lab = labels.get(int(atom_id), {})
        out.append(
            {
                "atom_id": int(atom_id),
                "coefficient": float(code[int(atom_id)]),
                "importance_percent": 100.0 * float(code[int(atom_id)]) / total,
                "label": lab.get("label", f"atom_{int(atom_id)}"),
                "description": lab.get("microscopy_description", ""),
                "confidence": lab.get("confidence", ""),
            }
        )
    return out


def faiss_neighbors(
    z: np.ndarray,
    query_row: int,
    df: pd.DataFrame,
    *,
    k: int = 8,
    excluded_terms: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    import faiss

    xb = np.ascontiguousarray(np.asarray(z, dtype=np.float32))
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    q = int(query_row) % int(xb.shape[0])
    normalized_exclusions = tuple(
        term.strip().casefold() for term in excluded_terms if term.strip()
    )
    search_k = xb.shape[0] if normalized_exclusions else min(int(k) + 1, xb.shape[0])
    dist2, idx = index.search(xb[q : q + 1], search_k)
    hits: list[dict[str, Any]] = []
    for ix, d2 in zip(idx[0].tolist(), dist2[0].tolist()):
        if ix < 0 or ix == q:
            continue
        row = df.iloc[int(ix)]
        composition = safe_text(row.get("composition", ""))
        hit = {
            "database_row": int(ix),
            "d2": float(d2),
            "specimen": safe_text(row.get("Specimen Name", "")),
            "source": safe_text(row.get("Source", "")),
            "illumination": safe_text(row.get("Illumination Modality", "")),
            "magnification": safe_text(row.get("Magnification", "")),
            "class": safe_text(row.get("Class", "")),
            "subclass": safe_text(row.get("Subclass", "")),
            "description": safe_text(row.get("Description", "")),
            "composition": composition,
            "chemical_formula": composition,
            "image": safe_text(row.get("Image", "")),
        }
        evidence_text = " ".join(safe_text(value) for value in hit.values()).casefold()
        if any(term in evidence_text for term in normalized_exclusions):
            continue
        hits.append(hit)
        if len(hits) >= int(k):
            break
    return hits


def faiss_neighbors_by_vector(
    z: np.ndarray,
    query_embedding: np.ndarray,
    df: pd.DataFrame,
    *,
    k: int = 10,
    excluded_terms: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Return corpus neighbors for an embedding that is not a database row."""
    import faiss

    xb = np.ascontiguousarray(np.asarray(z, dtype=np.float32))
    query = np.ascontiguousarray(
        np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    )
    if query.shape[1] != xb.shape[1]:
        raise ValueError(
            f"query embedding dim {query.shape[1]} != corpus dim {xb.shape[1]}"
        )
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    normalized_exclusions = tuple(
        term.strip().casefold() for term in excluded_terms if term.strip()
    )
    search_k = xb.shape[0] if normalized_exclusions else min(int(k), xb.shape[0])
    dist2, idx = index.search(query, search_k)
    hits: list[dict[str, Any]] = []
    for ix, d2 in zip(idx[0].tolist(), dist2[0].tolist()):
        if ix < 0:
            continue
        row = df.iloc[int(ix)]
        composition = safe_text(row.get("composition", ""))
        hit = {
            "database_row": int(ix),
            "d2": float(d2),
            "specimen": safe_text(row.get("Specimen Name", "")),
            "source": safe_text(row.get("Source", "")),
            "illumination": safe_text(row.get("Illumination Modality", "")),
            "magnification": safe_text(row.get("Magnification", "")),
            "class": safe_text(row.get("Class", "")),
            "subclass": safe_text(row.get("Subclass", "")),
            "description": safe_text(row.get("Description", "")),
            "composition": composition,
            "chemical_formula": composition,
            "image": safe_text(row.get("Image", "")),
        }
        evidence_text = " ".join(safe_text(value) for value in hit.values()).casefold()
        if any(term in evidence_text for term in normalized_exclusions):
            continue
        hits.append(hit)
        if len(hits) >= int(k):
            break
    return hits


def _l2_neighbor_consensus(
    neighbors: list[dict[str, Any]],
    *,
    identity_max_nearest_d2: float | None = None,
) -> dict[str, Any]:
    def rank_field(
        field: str,
        *,
        excluded: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        excluded = excluded or set()
        for rank, neighbor in enumerate(neighbors, start=1):
            value = safe_text(neighbor.get(field, "")).strip()
            key = re.sub(r"\s+", "", value).casefold()
            if not key or key in excluded:
                continue
            group = groups.setdefault(
                key,
                {
                    "value": value,
                    "count": 0,
                    "neighbor_ranks": [],
                    "database_rows": [],
                    "d2_values": [],
                },
            )
            group["count"] += 1
            group["neighbor_ranks"].append(rank)
            group["database_rows"].append(int(neighbor["database_row"]))
            group["d2_values"].append(float(neighbor["d2"]))
        ranked: list[dict[str, Any]] = []
        for group in groups.values():
            distances = group.pop("d2_values")
            group["mean_d2"] = float(sum(distances) / len(distances))
            ranked.append(group)
        return sorted(ranked, key=lambda item: (-int(item["count"]), item["mean_d2"]))

    specimens = rank_field("specimen")
    formulas = rank_field(
        "chemical_formula",
        excluded={"unknown", "unspecified", "variable", "n/a", "na", "none", "-"},
    )
    family_groups: dict[str, dict[str, Any]] = {}
    for rank, neighbor in enumerate(neighbors, start=1):
        specimen = safe_text(neighbor.get("specimen", "")).strip()
        normalized = re.sub(r"\s+", " ", specimen).casefold()
        if "ultramarine" in normalized or normalized == "lazurite":
            key = "ultramarine/lazurite"
            display = "ultramarine/lazurite"
        else:
            key = normalized
            display = specimen
        if not key:
            continue
        family = family_groups.setdefault(
            key,
            {
                "value": display,
                "count": 0,
                "neighbor_ranks": [],
                "database_rows": [],
                "member_names": [],
                "d2_values": [],
            },
        )
        family["count"] += 1
        family["neighbor_ranks"].append(rank)
        family["database_rows"].append(int(neighbor["database_row"]))
        family["d2_values"].append(float(neighbor["d2"]))
        if specimen and specimen not in family["member_names"]:
            family["member_names"].append(specimen)
    identity_families: list[dict[str, Any]] = []
    for family in family_groups.values():
        distances = family.pop("d2_values")
        family["mean_d2"] = float(sum(distances) / len(distances))
        identity_families.append(family)
    identity_families.sort(
        key=lambda item: (-int(item["count"]), float(item["mean_d2"]))
    )
    strong_family_threshold = max(3, int(np.ceil(0.8 * len(neighbors))))
    strong_identity_family = (
        identity_families[0]
        if identity_families
        and int(identity_families[0]["count"]) >= strong_family_threshold
        else None
    )
    most_repeated_specimen = (
        specimens[0] if specimens and int(specimens[0]["count"]) >= 2 else None
    )
    nearest_d2 = (
        min(float(neighbor["d2"]) for neighbor in neighbors)
        if neighbors
        else float("inf")
    )
    distance_identity_allowed = (
        identity_max_nearest_d2 is None
        or nearest_d2 <= float(identity_max_nearest_d2)
    )
    identity_allowed = distance_identity_allowed or strong_identity_family is not None
    repeated_formula = None
    if (
        identity_allowed
        and most_repeated_specimen is not None
        and formulas
        and int(formulas[0]["count"]) >= 3
    ):
        repeated_formula = formulas[0]
    return {
        "neighbor_count": len(neighbors),
        "selection_rule": (
            "distance or at least 80% same-family agreement; corpus frequency is not a "
            "prevalence prior"
        ),
        "nearest_d2": nearest_d2,
        "identity_max_nearest_d2": identity_max_nearest_d2,
        "distance_identity_allowed": distance_identity_allowed,
        "identity_inference_allowed": identity_allowed,
        "identity_family_frequencies": identity_families,
        "strong_identity_family_threshold": strong_family_threshold,
        "strong_identity_family_consensus": strong_identity_family,
        "specimen_frequencies": specimens,
        "most_repeated_specimen": (
            most_repeated_specimen if identity_allowed else None
        ),
        "formula_frequencies": formulas,
        "formula_repeat_threshold": 3,
        "formula_requires_repeated_specimen": True,
        "repeated_formula_suggestion": repeated_formula,
    }


def _fallback_sample_synthesis(
    *,
    illumination: str,
    recorded_magnification: str,
    neighbors: list[dict[str, Any]],
    l2_consensus: dict[str, Any],
    visual: dict[str, Any],
    top_atom: dict[str, Any],
    ranked_atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a conservative result when coordinator synthesis exhausts its retries."""

    morphology = visual.get("morphology") if isinstance(visual, dict) else {}
    if not isinstance(morphology, dict):
        morphology = {}
    primary = safe_text(morphology.get("primary", "")) or "microscopy specimen"

    illumination_text = safe_text(illumination)
    if illumination_text.lower() in {"", "unknown", "unspecified"}:
        illumination_text = "recorded microscopy"

    magnification_text = safe_text(recorded_magnification)
    if magnification_text.lower() in {"", "unknown", "unspecified"}:
        estimate = visual.get("magnification_estimate", {})
        if isinstance(estimate, dict):
            magnification_text = safe_text(estimate.get("estimate", ""))
    if not magnification_text:
        magnification_text = "estimated microscopy magnification"
    magnification_text = re.sub(
        r"(?i)(\d+(?:\.\d+)?)\s*x\b",
        lambda match: f"{match.group(1)}×",
        magnification_text,
    )

    sentences = [
        (
            f"Under {illumination_text} illumination at approximately "
            f"{magnification_text}, the field shows {primary} morphology "
            "[Vision]."
        )
    ]

    identity_family = l2_consensus.get("strong_identity_family_consensus")
    repeated_specimen = l2_consensus.get("most_repeated_specimen")
    if isinstance(identity_family, dict):
        name = safe_text(identity_family.get("value", ""))
        count = int(identity_family.get("count", 0))
        total = int(l2_consensus.get("neighbor_count", len(neighbors)))
        sentences.append(
            f"The nearest neighbors support {name} as the leading interpretation, with "
            f"{count} of {total} neighbors agreeing at the material-family level [FAISS]."
        )
    elif isinstance(repeated_specimen, dict):
        name = safe_text(repeated_specimen.get("value", ""))
        count = int(repeated_specimen.get("count", 0))
        total = int(l2_consensus.get("neighbor_count", len(neighbors)))
        sentences.append(
            f"The nearest neighbors retrieve {name} most frequently, with {count} of "
            f"{total} neighbors supporting it as the leading catalog identity [FAISS]."
        )
    elif neighbors:
        lead = neighbors[0]
        context = " ".join(
            part
            for part in (
                safe_text(lead.get("class", "")),
                safe_text(lead.get("subclass", "")),
            )
            if part
        )
        if not context:
            context = "the nearest catalog record"
        sentences.append(
            f"The nearest neighbors provide {context} as the primary catalog context "
            "[FAISS]."
        )

    repeated_formula = l2_consensus.get("repeated_formula_suggestion")
    suggested_formula = ""
    if isinstance(repeated_formula, dict):
        suggested_formula = safe_text(repeated_formula.get("value", ""))
        count = int(repeated_formula.get("count", 0))
        sentences.append(
            f"The chemical formula {suggested_formula} recurs in {count} nearest neighbors "
            "and is the suggested composition [FAISS]."
        )

    top_atom_id = int(top_atom["atom_id"])
    top_atom_label = safe_text(top_atom.get("label", f"atom {top_atom_id}"))
    sentences.append(
        f"The image interpretation includes {top_atom_label} as the leading atom-derived "
        f"feature [Atoms: {top_atom_id}]."
    )

    def first_positive(field: str) -> str:
        values = visual.get(field, []) if isinstance(visual, dict) else []
        if not isinstance(values, list):
            return ""
        for value in values:
            text = safe_text(value).rstrip(".")
            if text and not _NEGATIVE_CONTRAST_RE.search(text):
                return text
        return ""

    visual_fields = [
        ("color_and_tone", "color and tone"),
        ("size_and_distribution", "size and distribution"),
        ("optical_cues", "optical response"),
    ]
    formal_elements: dict[str, str] = {}
    for field, label in visual_fields:
        observation = first_positive(field)
        if observation:
            sentences.append(
                f"The {label} evidence shows {observation[0].lower() + observation[1:]} "
                "[Vision]."
            )
            formal_elements[label.replace(" ", "_")] = observation

    morphology_notes = morphology.get("notes", [])
    if isinstance(morphology_notes, list):
        for note in morphology_notes:
            text = safe_text(note).rstrip(".")
            if text and not _NEGATIVE_CONTRAST_RE.search(text):
                sentences.append(
                    f"The morphology includes {text[0].lower() + text[1:]} [Vision]."
                )
                break

    fallback_context = [
        (
            "The visual observations provide the direct account of morphology and "
            "optical response [Vision]."
        ),
        (
            "The nearest neighbors supply the primary catalog context for this "
            "interpretation [FAISS]."
        ),
        (
            "The combined evidence supports a microscopy description grounded in the "
            "retrieved records and visible field [FAISS] [Vision]."
        ),
        (
            "The catalog context and image features jointly support the stated sample "
            "interpretation [FAISS] [Vision]."
        ),
    ]
    for filler in fallback_context:
        if len(sentences) >= 6:
            break
        if filler not in sentences:
            sentences.append(filler)

    formal_elements.setdefault("morphology", primary)
    fallback_atom_contributions: list[dict[str, Any]] = []
    for atom in ranked_atoms[:10]:
        atom_id = int(atom["atom_id"])
        atom_label = safe_text(atom.get("label", "")) or f"atom {atom_id}"
        fallback_atom_contributions.append(
            {
                "atom_id": atom_id,
                "importance_percent": float(atom.get("importance_percent", 0.0)),
                "contribution": (
                    top_atom_label
                    if atom_id == top_atom_id
                    else (
                        "Considered for transferable microscopy attributes represented by "
                        f"{atom_label}."
                    )
                ),
            }
        )
    return {
        "microscopy_description": " ".join(sentences),
        "established_type": primary,
        "suggested_chemical_formula": suggested_formula,
        "magnification_estimate": magnification_text,
        "formal_elements": formal_elements,
        "uncertainties": [],
        "atom_contributions": fallback_atom_contributions,
        "fallback_used": True,
    }


def _coordinator_json(
    prompt: str,
    stage: str,
    *,
    backend: str,
    model: str,
    validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    active_prompt = prompt
    for attempt in range(5):
        try:
            raw = _answer_chat(
                active_prompt,
                backend=backend,
                model=model,
                max_tokens=8192,
                temperature=0.1,
            )
            payload = extract_json_object(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            if validator is not None:
                validator(payload)
            progress(f"DONE {stage}")
            return payload
        except Exception as exc:
            last_error = exc
            progress(f"FAIL {stage} attempt {attempt+1}/5: {exc}")
            if validator is not None and (
                "positive-evidence violation" in str(exc)
                or "duplicate atom citation violation" in str(exc)
                or "opening context violation" in str(exc)
                or "L2 consensus violation" in str(exc)
                or "top atom violation" in str(exc)
                or "neighbor wording violation" in str(exc)
                or "retrieval confidence violation" in str(exc)
                or "identity family violation" in str(exc)
                or "atom consideration violation" in str(exc)
            ):
                active_prompt = (
                    prompt
                    + "\n\nCORRECTION AFTER REJECTED DRAFT\n"
                    + f"The prior draft was rejected: {exc}\n"
                    + "Rewrite the entire JSON using affirmative evidence only. Remove the "
                    + "offending clause completely; do not paraphrase it as another denial "
                    + "or comparison. In atom_contributions, state only the positive visual "
                    + "attribute transferred to this sample. Cite every atom ID at most once "
                    + "across the entire microscopy_description. Cite only relevant atoms. "
                    + "Include at least 10 distinct supplied atoms in atom_contributions, "
                    + "even though only prose-relevant atoms need citations. "
                    + "Put illumination in sentence 1 and magnification in sentence "
                    + "1 or 2. Follow the L2 frequency consensus and include its repeated "
                    + "chemical formula suggestion when supplied. Use the exact top-ranked "
                    + "atom label phrase and cite that atom exactly once. Write \"the nearest "
                    + "neighbors,\" never \"the L2 neighborhood.\" When retrieval identity "
                    + "confidence is false, leave suggested_chemical_formula empty and treat "
                    + "neighbor identities as visual analogs only.\n"
                )
            time.sleep(min(30.0, 2.0 * (2**attempt)))
    raise RuntimeError(f"{stage} failed: {last_error}")


def _coordinator_vision_json(
    prompt: str,
    image_path: Path,
    *,
    backend: str,
    model: str,
    validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            raw = chat_vision(
                prompt,
                image_path,
                backend=backend,
                model=model,
                max_tokens=4096,
                temperature=0.1,
            )
            payload = extract_json_object(raw)
            if not isinstance(payload, dict):
                raise ValueError("Vision response was not a JSON object")
            if validator is not None:
                validator(payload)
            return payload
        except Exception as exc:
            last_error = exc
            progress(f"Vision fail attempt {attempt+1}/5 ({backend}): {exc}")
            time.sleep(min(30.0, 2.0 * (2**attempt)))
    raise RuntimeError(f"Vision observation failed: {last_error}")


def describe_sample(
    *,
    query_row: int,
    z: np.ndarray,
    df: pd.DataFrame,
    codes: np.ndarray,
    labels: dict[int, dict[str, Any]],
    images_dir: Path,
    out_path: Path,
    cfg: ClassicalKsvdConfig,
    coordinator_backend: str = "openai",
    coordinator_model: str = COORDINATOR_MODEL_DEFAULT,
    faiss_k: int = 8,
    identity_max_nearest_d2: float | None = None,
    excluded_neighbor_terms: tuple[str, ...] = (),
) -> dict[str, Any]:
    load_env()

    q = int(query_row) % int(z.shape[0])
    row = df.iloc[q]
    image_name = safe_text(row.get("Image", ""))
    image_path = images_dir / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image for row {q}: {image_path}")

    all_active_atoms = active_weighted_atoms(
        codes[q],
        labels,
        active_eps=float(cfg.active_eps),
        top_n=None,
    )
    atoms = [
        atom
        for atom in all_active_atoms
        if float(atom["importance_percent"]) >= 1.0
    ]
    if len(atoms) < 10:
        included_ids = {int(atom["atom_id"]) for atom in atoms}
        atoms.extend(
            atom
            for atom in all_active_atoms
            if int(atom["atom_id"]) not in included_ids
        )
        atoms = atoms[:10]
    neighbors = faiss_neighbors(
        z,
        q,
        df,
        k=faiss_k,
        excluded_terms=excluded_neighbor_terms,
    )
    l2_consensus = _l2_neighbor_consensus(
        neighbors,
        identity_max_nearest_d2=identity_max_nearest_d2,
    )
    repeated_formula = l2_consensus.get("repeated_formula_suggestion")
    illumination = safe_text(row.get("Illumination Modality", ""))
    recorded_magnification = safe_text(row.get("Magnification", ""))
    target = {
        "database_row": q,
        "image": image_name,
        "specimen": safe_text(row.get("Specimen Name", "")),
        "source": safe_text(row.get("Source", "")),
        "illumination_modality": illumination,
        "recorded_magnification": recorded_magnification,
    }
    # Blind target for synthesis prompts (avoid leaking identity into atom prose).
    blind_target = {
        "database_row": q,
        "source": "database row",
        "illumination_modality": illumination,
        "recorded_magnification": recorded_magnification,
    }

    progress(f"Sample row {q}: nearest neighbors (primary), then vision, then atoms")
    vision_fallback_error = ""
    try:
        visual = _coordinator_vision_json(
            VISUAL_OBSERVATION_PROMPT,
            image_path,
            backend=coordinator_backend,
            model=coordinator_model,
            validator=_validate_positive_visual_prose,
        )
    except Exception as exc:
        vision_fallback_error = str(exc)
        progress(
            "WARN direct vision exhausted retries; using nearest-neighbor context "
            "for synthesis"
        )
        visual = {
            "morphology": {
                "primary": "microscopy sample",
                "confidence": "low",
                "notes": [],
            },
            "color_and_tone": [],
            "size_and_distribution": [],
            "magnification_estimate": {
                "estimate": recorded_magnification or "estimated magnification",
                "confidence": "low",
                "basis": "nearest-neighbor microscopy context",
            },
            "optical_cues": [],
            "positive_constraints": {
                "supported": ["microscopy image interpretation"],
                "summary": "Nearest-neighbor evidence supplies the primary sample context.",
            },
            "uncertainties": [],
        }

    # Compact type hint for atom rerank (no query specimen name).
    neighbor_types: list[str] = []
    for h in neighbors:
        bits = [h.get("class", ""), h.get("subclass", ""), h.get("specimen", "")]
        neighbor_types.append(" | ".join(b for b in bits if b))
    morph = visual.get("morphology") if isinstance(visual, dict) else {}
    if not isinstance(morph, dict):
        morph = {}
    type_hint = {
        "vision_primary": morph.get("primary", "uncertain"),
        "vision_confidence": morph.get("confidence", ""),
        "vision_notes": morph.get("notes", []),
        "faiss_neighbor_summaries": neighbor_types,
        "faiss_frequency_consensus": l2_consensus,
    }

    atom_text = json.dumps(atoms, ensure_ascii=False, indent=2)
    rerank_fallback_error = ""
    try:
        rerank = _coordinator_json(
            SAMPLE_RERANK_PROMPT.format(
                target=json.dumps(blind_target, ensure_ascii=False, indent=2),
                type_hint=json.dumps(type_hint, ensure_ascii=False, indent=2),
                atoms=atom_text,
            ),
            "sample atom rerank",
            backend=coordinator_backend,
            model=coordinator_model,
        )
    except Exception as exc:
        rerank_fallback_error = str(exc)
        progress("WARN sample atom rerank exhausted retries; using coefficient order")
        rerank = {
            "reranked_atom_ids": [int(atom["atom_id"]) for atom in atoms],
            "rationale": "Coefficient order fallback.",
            "fallback_used": True,
        }
    ranked_ids = [int(x) for x in rerank.get("reranked_atom_ids", [])]
    by_id = {int(a["atom_id"]): a for a in atoms}
    if sorted(ranked_ids) != sorted(by_id):
        ranked_atoms = atoms
    else:
        ranked_atoms = [by_id[i] for i in ranked_ids]
    required_atoms = ranked_atoms
    top_atom = ranked_atoms[0]

    def validate_sample_synthesis(payload: dict[str, Any]) -> None:
        _validate_positive_sample_prose(payload)
        description = str(payload.get("microscopy_description", ""))
        _validate_neighbor_wording(description)
        _validate_top_atom_use(description, top_atom)
        _validate_opening_context(
            description,
            illumination=illumination,
            magnification=recorded_magnification,
        )
        _validate_repeated_formula_suggestion(payload, repeated_formula)
        _validate_retrieval_confidence(payload, l2_consensus)
        _validate_identity_family_consensus(payload, l2_consensus)
        _validate_minimum_atom_contributions(payload, ranked_atoms, minimum=10)

    synthesis_fallback_error = ""
    try:
        synthesis = _coordinator_json(
            SAMPLE_SYNTHESIS_PROMPT.format(
                target=json.dumps(blind_target, ensure_ascii=False, indent=2),
                neighbors=json.dumps(neighbors, ensure_ascii=False, indent=2),
                l2_consensus=json.dumps(l2_consensus, ensure_ascii=False, indent=2),
                visual_observations=json.dumps(visual, ensure_ascii=False, indent=2),
                atoms=json.dumps(ranked_atoms, ensure_ascii=False, indent=2),
                top_atom=json.dumps(top_atom, ensure_ascii=False, indent=2),
                required_high_weight_atoms=json.dumps(
                    required_atoms,
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
            "sample microscopy synthesis",
            backend=coordinator_backend,
            model=coordinator_model,
            validator=validate_sample_synthesis,
        )
    except Exception as exc:
        synthesis_fallback_error = str(exc)
        progress(
            "WARN sample microscopy synthesis exhausted retries; "
            "using deterministic L2 + vision fallback"
        )
        synthesis = _fallback_sample_synthesis(
            illumination=illumination,
            recorded_magnification=recorded_magnification,
            neighbors=neighbors,
            l2_consensus=l2_consensus,
            visual=visual,
            top_atom=top_atom,
            ranked_atoms=ranked_atoms,
        )

    output = {
        "target": target,
        "pipeline": {
            "dictionary_atoms": int(codes.shape[1]),
            "active_eps": float(cfg.active_eps),
            "elastic_net_alpha": float(cfg.elastic_net_alpha),
            "elastic_net_l1_ratio": float(cfg.elastic_net_l1_ratio),
            "sparse_codes_nonnegative": bool(cfg.sparse_codes_nonnegative),
            "coordinator_backend": coordinator_backend,
            "coordinator_model": coordinator_model,
            "faiss_k": int(faiss_k),
            "atom_description_min_percent": 1.0,
            "minimum_considered_atom_count": 10,
            "active_atom_count": len(all_active_atoms),
            "included_atom_count": len(ranked_atoms),
            "evidence_priority": [
                "faiss_l2_type_identity",
                "vision_image_appearance",
                "dictionary_atoms_critical_interpret",
            ],
            "positive_evidence_only_after_type": True,
            "atoms_require_interpretation": True,
            "vision_fallback_used": bool(vision_fallback_error),
            "vision_fallback_error": vision_fallback_error,
            "atom_rerank_fallback_used": bool(rerank_fallback_error),
            "atom_rerank_fallback_error": rerank_fallback_error,
            "synthesis_fallback_used": bool(synthesis.get("fallback_used", False)),
            "synthesis_fallback_error": synthesis_fallback_error,
        },
        "type_hint": type_hint,
        "faiss_neighbors": neighbors,
        "faiss_l2_consensus": l2_consensus,
        "direct_visual_observations": visual,
        "weighted_atoms": ranked_atoms,
        "coordinator_rerank": rerank,
        "established_type": synthesis.get("established_type", morph.get("primary", "")),
        "suggested_chemical_formula": synthesis.get("suggested_chemical_formula", ""),
        "magnification_estimate": synthesis.get("magnification_estimate", ""),
        "microscopy_description": synthesis.get("microscopy_description", ""),
        "formal_elements": synthesis.get("formal_elements", {}),
        "uncertainties": synthesis.get("uncertainties", []),
        "atom_contributions": synthesis.get("atom_contributions", []),
        "raw_synthesis": synthesis,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    progress(f"Wrote sample description → {out_path}")
    return output
