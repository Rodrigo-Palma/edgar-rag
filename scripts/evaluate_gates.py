"""Compare the two relevance gates on questions whose answer we know.

The interesting failure is not the question the filing obviously covers, it is
the question that shares vocabulary with the filing while not being in it. A
cosine gate cannot tell those apart by construction: it only measures that the
words are nearby. This script measures whether the model gate can.
"""

import argparse
from dataclasses import dataclass

from edgar_rag.config import EvalSettings
from edgar_rag.domain import DEFAULT_TOP_K
from edgar_rag.embeddings import OllamaEmbedder
from edgar_rag.gate import BrierGate, CosineGate, GateError, RelevanceGate
from edgar_rag.index import FilingIndex


@dataclass(frozen=True, slots=True)
class Case:
    """A question and whether this filing can answer it."""

    question: str
    answerable: bool
    note: str = ""


CASES = (
    Case("what products does the company design and sell?", True),
    Case("what are the risks to the company's supply chain?", True),
    Case("how much did the company spend on research and development?", True),
    Case("what does the company say about its share repurchase program?", True),
    Case("in which state is the company incorporated?", True),
    Case("who won the football league in 1998?", False, "nothing to do with the filing"),
    Case(
        "what were the company's revenues in 1994?",
        False,
        "same vocabulary as the filing, wrong period",
    ),
    Case(
        "what is the company's dividend policy for 2031?",
        False,
        "same vocabulary, a year the filing cannot cover",
    ),
    Case(
        "how many employees does Petrobras have?",
        False,
        "a filing-shaped question about another company",
    ),
    Case(
        "what does the company say about its plans to enter the airline business?",
        False,
        "filing vocabulary, a business it is not in",
    ),
)


@dataclass(frozen=True, slots=True)
class Score:
    """Counts, so the trade-off is visible instead of averaged away."""

    name: str
    admitted_answerable: int
    total_answerable: int
    admitted_unanswerable: int
    total_unanswerable: int

    @property
    def recall(self) -> float:
        return self.admitted_answerable / self.total_answerable

    @property
    def false_admissions(self) -> float:
        return self.admitted_unanswerable / self.total_unanswerable


def run(gate: RelevanceGate, name: str, index: FilingIndex, embedder) -> Score:
    admitted = {True: 0, False: 0}
    print(f"\n{name}")
    for case in CASES:
        passages = index.search(embedder.embed((case.question,)), top_k=DEFAULT_TOP_K)
        try:
            decision = gate.admits(case.question, passages)
        except GateError as error:
            print(f"  gate unavailable: {error}")
            raise
        admitted[case.answerable] += int(decision.admitted)
        verdict = "answer " if decision.admitted else "abstain"
        correct = " " if decision.admitted == case.answerable else "X"
        print(f"  {correct} {verdict}  {case.question[:52]:<52} {decision.reason[:58]}")

    return Score(
        name=name,
        admitted_answerable=admitted[True],
        total_answerable=sum(case.answerable for case in CASES),
        admitted_unanswerable=admitted[False],
        total_unanswerable=sum(not case.answerable for case in CASES),
    )


def sweep(name: str, scores: dict[str, float], grid: tuple[float, ...]) -> None:
    """Both gates over their whole threshold range, compared at equal recall.

    Comparing one gate at 0.55 against another at 0.7 compares two arbitrary
    points, not two detectors: whichever is more conservative looks better at
    rejecting and worse at answering. The honest question is what each one
    rejects when both answer the same share of the answerable questions.
    """
    print(f"\n{name}, threshold sweep:")
    print("    threshold   answers answerable   wrongly answers the rest")
    for threshold in grid:
        admitted_yes = sum(
            1 for case in CASES if case.answerable and scores[case.question] >= threshold
        )
        admitted_no = sum(
            1 for case in CASES if not case.answerable and scores[case.question] >= threshold
        )
        print(f"      {threshold:.2f}        {admitted_yes}/5                  {admitted_no}/5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brier-url", default="http://localhost:8100")
    parser.add_argument("--brier-confidence", type=float, default=0.7)
    arguments = parser.parse_args()

    settings = EvalSettings()
    index = FilingIndex.load(settings.index_dir)
    embedder = OllamaEmbedder(str(settings.ollama_base_url), settings.embedding_model)
    print(f"filing: {index.source['company']} {index.source['form']}, {len(index.chunks)} chunks")

    scores = [
        run(
            CosineGate(settings.min_retrieval_score),
            f"cosine >= {settings.min_retrieval_score}",
            index,
            embedder,
        ),
        run(
            BrierGate(arguments.brier_url, min_confidence=arguments.brier_confidence),
            f"brier >= {arguments.brier_confidence}",
            index,
            embedder,
        ),
    ]

    header = f"{'gate':<22} {'answers the answerable':<24} {'wrongly answers the rest'}"
    print(f"\n{header}")
    for score in scores:
        print(
            f"{score.name:<22} "
            f"{score.admitted_answerable}/{score.total_answerable} ({score.recall:.0%})"
            f"{'':<12} "
            f"{score.admitted_unanswerable}/{score.total_unanswerable} "
            f"({score.false_admissions:.0%})"
        )
    _sweeps(index, embedder, arguments)
    return 0


def _sweeps(index: FilingIndex, embedder, arguments) -> None:
    """Collect the raw score each gate assigns, then sweep both."""
    cosine_scores, brier_scores = {}, {}
    brier = BrierGate(arguments.brier_url, min_confidence=0.0)

    for case in CASES:
        passages = index.search(embedder.embed((case.question,)), top_k=DEFAULT_TOP_K)
        cosine_scores[case.question] = passages[0].score if passages else 0.0
        try:
            best = max(
                brier.confidence_for(case.question, scored.chunk.text) for scored in passages
            )
        except GateError as error:
            print(f"\nbrier unavailable, skipping the sweep: {error}")
            return
        brier_scores[case.question] = best

    sweep("cosine", cosine_scores, (0.45, 0.50, 0.55, 0.60, 0.65, 0.70))
    sweep("brier", brier_scores, (0.2, 0.4, 0.5, 0.6, 0.7, 0.8))


if __name__ == "__main__":
    raise SystemExit(main())
