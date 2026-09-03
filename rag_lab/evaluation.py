"""
Stage 9: Evaluation.

Compares semantic retrieval against reranked retrieval on a small
hand-built evaluation set, using the two standard IR metrics:

- Top-1 Accuracy: is the single highest-ranked chunk from the
  document that should have answered the question?
- MRR (Mean Reciprocal Rank): 1 / (rank of the first correct chunk),
  averaged across questions — rewards getting the right answer near
  the top even when it isn't rank 1.

This is deliberately a small, hand-labeled set (not a held-out split
of the corpus) — the goal is to demonstrate the evaluation
methodology and see semantic vs. reranked actually differ, not to
produce a statistically rigorous benchmark from four short policy
documents.

Run directly: python -m rag_lab.evaluation
"""
from rag_lab.chunking import chunk_documents
from rag_lab.ingestion import load_documents
from rag_lab.reranking import rerank
from rag_lab.retrieval import SemanticRetriever

EVAL_SET = [

    {
        "question": "How many casual leave days can I take in a year?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "Do I need a medical certificate for sick leave?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "When does a new employee become eligible for annual leave?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "How many days of annual leave am I entitled to per year?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "How many annual leave days carry forward to next year?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "What happens to unused sick leave at year end?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "Am I paid out for unused leave when I leave the company?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "By when must carried-forward annual leave be used?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "What public holidays does the company observe?",
        "expected_title": "Holiday Calendar",
    },

    {
        "question": "Are weekends already accounted for in the holiday calendar?",
        "expected_title": "Holiday Calendar",
    },

    {
        "question": "How long does a manager have to approve a leave request?",
        "expected_title": "Approval Escalation",
    },

    {
        "question": "Can I cancel an already-approved leave request?",
        "expected_title": "Approval Escalation",
    },

]

HARD_EVAL_SET = [

    # ------------------------------------------------------------
    # LEAVE TYPES / ELIGIBILITY
    # ------------------------------------------------------------

    {
        "question": "How much short-notice personal leave am I allowed each year?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "After joining the company, how long do I have to wait before I can use annual leave?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "Is sick leave available immediately when someone joins?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "If I suddenly need a day off for a personal matter, which leave category is intended for that?",
        "expected_title": "Leave Types And Eligibility",
    },

    # ------------------------------------------------------------
    # CARRY FORWARD
    # ------------------------------------------------------------

    {
        "question": "What happens to unused days at the end of the year if they are from the casual category?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "Is there a limit on how much annual leave can survive into the following year?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "How long do I have to use annual leave that was brought forward from last year?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "When an employee leaves the company, which unused leave can be paid out?",
        "expected_title": "Carry Forward Rules",
    },

    # ------------------------------------------------------------
    # HOLIDAY CALENDAR
    # ------------------------------------------------------------

    {
        "question": "Do weekends reduce the number of leave days that get deducted?",
        "expected_title": "Holiday Calendar",
    },

    {
        "question": "If a company holiday falls during my planned leave, is that day treated like a leave day?",
        "expected_title": "Holiday Calendar",
    },

    {
        "question": "Where can I find the company's list of public holidays?",
        "expected_title": "Holiday Calendar",
    },

    # ------------------------------------------------------------
    # APPROVAL / ESCALATION
    # ------------------------------------------------------------

    {
        "question": "What happens when my leave request sits with my manager without a response?",
        "expected_title": "Approval Escalation",
    },

    {
        "question": "Who should I contact when a leave request stays pending for too long?",
        "expected_title": "Approval Escalation",
    },

    {
        "question": "Can a leave request be withdrawn after the manager has already approved it?",
        "expected_title": "Approval Escalation",
    },

    {
        "question": "What is the process when a manager does not act on a leave request?",
        "expected_title": "Approval Escalation",
    },

    # ------------------------------------------------------------
    # INDIRECT / PARAPHRASED QUESTIONS
    # ------------------------------------------------------------

    {
        "question": "I have some days left over from this year. Which types disappear and which can move to next year?",
        "expected_title": "Carry Forward Rules",
    },

    {
        "question": "What is the company's rule for taking time off on Saturday or Sunday?",
        "expected_title": "Holiday Calendar",
    },

    {
        "question": "How much notice is normally expected before planned annual time off?",
        "expected_title": "Leave Types And Eligibility",
    },

    {
        "question": "What documentation might HR need after several consecutive sick days?",
        "expected_title": "Leave Types And Eligibility",
    },
]

def top1_hit(ranked_titles: list[str], expected_title: str) -> int:
    return 1 if ranked_titles and ranked_titles[0] == expected_title else 0


def reciprocal_rank(ranked_titles: list[str], expected_title: str) -> float:
    for rank, title in enumerate(ranked_titles, start=1):
        if title == expected_title:
            return 1.0 / rank
    return 0.0


def evaluate(documents_folder: str = "rag_lab/documents", k: int = 5) -> dict:
    documents = load_documents(documents_folder)
    chunks = chunk_documents(documents)
    retriever = SemanticRetriever(chunks)

    semantic_top1 = semantic_rr = reranked_top1 = reranked_rr = 0.0
    total = len(EVAL_SET)

    for item in EVAL_SET:
        question, expected = item["question"], item["expected_title"]

        semantic_results = retriever.retrieve(question, top_k=k)
        semantic_titles = [c["title"] for c, _ in semantic_results]
        semantic_top1 += top1_hit(semantic_titles, expected)
        semantic_rr += reciprocal_rank(semantic_titles, expected)

        reranked_results = rerank(question, semantic_results, top_n=k)
        reranked_titles = [c["title"] for c, _ in reranked_results]
        reranked_top1 += top1_hit(reranked_titles, expected)
        reranked_rr += reciprocal_rank(reranked_titles, expected)

    return {
        "total": total,
        "semantic_top1": semantic_top1 / total,
        "semantic_mrr": semantic_rr / total,
        "reranked_top1": reranked_top1 / total,
        "reranked_mrr": reranked_rr / total,
    }

def evaluate_hard(
    documents_folder: str = "rag_lab/documents",
    k: int = 5,
) -> dict:
    documents = load_documents(documents_folder)
    chunks = chunk_documents(documents)
    retriever = SemanticRetriever(chunks)

    semantic_top1 = 0.0
    semantic_rr = 0.0
    reranked_top1 = 0.0
    reranked_rr = 0.0

    total = len(HARD_EVAL_SET)

    print("\n" + "=" * 90)
    print("HARD RAG EVALUATION")
    print("=" * 90)

    for index, item in enumerate(HARD_EVAL_SET, start=1):

        question = item["question"]
        expected = item["expected_title"]

        semantic_results = retriever.retrieve(
            question,
            top_k=k,
        )

        semantic_titles = [
            chunk["title"]
            for chunk, _ in semantic_results
        ]

        semantic_rank = (
            semantic_titles.index(expected) + 1
            if expected in semantic_titles
            else None
        )

        reranked_results = rerank(
            question,
            semantic_results,
            top_n=k,
        )

        reranked_titles = [
            chunk["title"]
            for chunk, _ in reranked_results
        ]

        reranked_rank = (
            reranked_titles.index(expected) + 1
            if expected in reranked_titles
            else None
        )

        semantic_hit = top1_hit(
            semantic_titles,
            expected,
        )

        semantic_reciprocal = reciprocal_rank(
            semantic_titles,
            expected,
        )

        reranked_hit = top1_hit(
            reranked_titles,
            expected,
        )

        reranked_reciprocal = reciprocal_rank(
            reranked_titles,
            expected,
        )

        semantic_top1 += semantic_hit
        semantic_rr += semantic_reciprocal
        reranked_top1 += reranked_hit
        reranked_rr += reranked_reciprocal

        print("-" * 90)
        print(f"{index}. {question}")
        print(f"Expected : {expected}")
        print(
            f"Semantic : rank={semantic_rank}, "
            f"top1={'YES' if semantic_hit else 'NO'}"
        )
        print(
            f"Reranked : rank={reranked_rank}, "
            f"top1={'YES' if reranked_hit else 'NO'}"
        )

        if (
            semantic_rank is not None
            and reranked_rank is not None
            and reranked_rank < semantic_rank
        ):
            print(
                f"Movement: UP {semantic_rank - reranked_rank} position(s)"
            )
        elif (
            semantic_rank is not None
            and reranked_rank is not None
            and reranked_rank > semantic_rank
        ):
            print(
                f"Movement: DOWN {reranked_rank - semantic_rank} position(s)"
            )
        else:
            print("Movement: NO CHANGE")

    results = {
        "total": total,
        "semantic_top1": semantic_top1 / total,
        "semantic_mrr": semantic_rr / total,
        "reranked_top1": reranked_top1 / total,
        "reranked_mrr": reranked_rr / total,
    }

    print("\n" + "=" * 90)
    print("HARD EVALUATION SUMMARY")
    print("=" * 90)

    print(
        f"Semantic Top-1 Accuracy : "
        f"{semantic_top1:.0f}/{total} "
        f"({results['semantic_top1'] * 100:.1f}%)"
    )

    print(
        f"Reranked Top-1 Accuracy : "
        f"{reranked_top1:.0f}/{total} "
        f"({results['reranked_top1'] * 100:.1f}%)"
    )

    print(
        f"Semantic MRR            : "
        f"{results['semantic_mrr']:.3f}"
    )

    print(
        f"Reranked MRR            : "
        f"{results['reranked_mrr']:.3f}"
    )

    print(
        f"Top-1 improvement       : "
        f"{(results['reranked_top1'] - results['semantic_top1']) * 100:+.1f} pp"
    )

    print(
        f"MRR improvement         : "
        f"{results['reranked_mrr'] - results['semantic_mrr']:+.3f}"
    )

    return results

def print_results(results: dict) -> None:
    total = results["total"]
    print(f"Evaluated over {total} questions:\n")
    print(f"  Semantic Top-1 Accuracy: {results['semantic_top1']:.1%}")
    print(f"  Reranked Top-1 Accuracy: {results['reranked_top1']:.1%}")
    print(f"  Semantic MRR:            {results['semantic_mrr']:.3f}")
    print(f"  Reranked MRR:            {results['reranked_mrr']:.3f}")



if __name__ == "__main__":
    evaluate_hard()