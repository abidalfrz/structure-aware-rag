import json
import re
import time
import string
import pandas as pd
from collections import Counter
from statistics import mean
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

EVAL_PATH = "./evaluation_dataset.json"
OUTPUT_CSV = "./ablation_k_results.csv"
OUTPUT_JSON = "./ablation_k_results.json"

K_RANGE = [1, 2, 3, 4, 5]

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    evaluation_file = json.load(f)

if isinstance(evaluation_file, dict) and "items" in evaluation_file:
    evaluation_data = evaluation_file["items"]
else:
    evaluation_data = evaluation_file

model = OllamaLLM(model="qwen2.5:3b", temperature=0.0)

template = """
Anda adalah asisten resmi untuk kompetisi Sebelas Maret Statistics Data Science (SSDS) 2026.

Gunakan HANYA informasi dari konteks berikut.

<context>
{context}
</context>

Pertanyaan:
{question}

Instruksi:
1. Jawab hanya berdasarkan informasi pada konteks.
2. Jangan menggunakan pengetahuan di luar konteks.
3. Jangan membuat atau menambahkan informasi yang tidak tersedia.
4. Jika informasi yang ditanyakan tidak tersedia dalam konteks, jawab:
   "Maaf, informasi tersebut tidak tercantum dalam Buku Panduan."
5. Jawab secara ringkas tetapi sertakan semua syarat, batasan, nilai, tanggal, atau pengecualian yang secara langsung diperlukan untuk menjawab pertanyaan.
6. Jika konteks hanya memberikan informasi umum tetapi tidak memberikan nilai/detail spesifik yang diminta, nyatakan dengan jelas bahwa detail tersebut tidak tercantum. Jangan mengubah informasi umum menjadi jawaban spesifik.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text.strip()


def normalize_section(text):
    text = normalize_text(text)
    text = re.sub(r'\bsub bab\b', '', text)
    text = re.sub(r'\bbab\b', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def token_f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def lcs_length(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0

    previous = [0] * (len(tokens_b) + 1)

    for token_a in tokens_a:
        current = [0]

        for j, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(current[-1], previous[j]))

        previous = current

    return previous[-1]


def rouge_l_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, gt_tokens)

    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def phrase_in_text(phrase, text):
    phrase = normalize_text(phrase)
    text = normalize_text(text)

    return phrase in text


def must_include_recall(answer, must_include):
    if not must_include:
        return None

    matched = sum(
        1 for phrase in must_include
        if phrase_in_text(phrase, answer)
    )

    return matched / len(must_include)


def abstain_answer(answer):
    normalized_answer = normalize_text(answer)

    abstain_patterns = [
        "maaf informasi tersebut tidak tercantum dalam buku panduan",
        "informasi tersebut tidak tercantum",
        "informasi tidak tercantum",
        "tidak tercantum dalam buku panduan",
        "tidak terdapat dalam buku panduan",
        "tidak tersedia dalam buku panduan",
        "tidak terdapat dalam konteks",
        "tidak tersedia dalam konteks",
        "saya tidak tahu",
        "tidak tahu"
    ]

    return any(
        pattern in normalized_answer
        for pattern in abstain_patterns
    )


def document_matches_section(doc, gold_section):
    if not gold_section:
        return False

    gold_section = normalize_section(gold_section)

    bab_utama = normalize_section(
        doc.metadata.get("Bab Utama", "")
    )

    sub_bab = normalize_section(
        doc.metadata.get("Sub Bab", "")
    )

    if gold_section == bab_utama:
        return True

    if gold_section == sub_bab:
        return True

    if gold_section in bab_utama:
        return True

    if gold_section in sub_bab:
        return True

    return False


def recall_at_k(docs, gold_section, k):
    if not gold_section:
        return None

    docs_at_k = docs[:k]

    return float(
        any(
            document_matches_section(doc, gold_section)
            for doc in docs_at_k
        )
    )


def evidence_recall_at_k(docs, gold_evidence, k):
    if not gold_evidence:
        return None

    docs_at_k = docs[:k]

    combined_context = "\n".join(
        doc.page_content
        for doc in docs_at_k
    )

    matched = sum(
        1 for evidence in gold_evidence
        if phrase_in_text(evidence, combined_context)
    )

    return matched / len(gold_evidence)


def reciprocal_rank(docs, gold_section):
    if not gold_section:
        return None

    for rank, doc in enumerate(docs, start=1):
        if document_matches_section(doc, gold_section):
            return 1.0 / rank

    return 0.0


def first_relevant_rank(docs, gold_section):
    if not gold_section:
        return None

    for rank, doc in enumerate(docs, start=1):
        if document_matches_section(doc, gold_section):
            return rank

    return None


def safe_mean(values):
    values = [
        value for value in values
        if value is not None
    ]

    if not values:
        return None

    return mean(values)


def evaluate_with_k(evaluation_data, k):
    results = []

    for idx, sample in enumerate(evaluation_data, start=1):
        question_id = sample["id"]
        question = sample["question"]
        category = sample["category"]
        question_type = sample.get("question_type", "unknown")
        difficulty = sample.get("difficulty", "unknown")
        answerable = sample["answerable"]

        expected_answer = sample.get("expected_answer", "")
        must_include = sample.get("must_include", [])
        gold_section = sample.get("gold_section")
        gold_subsection = sample.get("gold_subsection")
        gold_evidence = sample.get("gold_evidence_contains", [])

        print("\n" + "=" * 100)
        print(f"K={k} | Evaluating {idx}/{len(evaluation_data)} | ID: {question_id}")
        print(f"Question: {question}")

        retrieval_start = time.perf_counter()

        all_docs = retriever.invoke(question)

        retrieval_end = time.perf_counter()

        retrieval_latency = retrieval_end - retrieval_start

        docs = all_docs[:k]

        recall_k = recall_at_k(
            all_docs,
            gold_section,
            k
        )

        evidence_recall_k = evidence_recall_at_k(
            all_docs,
            gold_evidence,
            k
        )

        rr = reciprocal_rank(
            all_docs,
            gold_section
        )

        first_rank = first_relevant_rank(
            all_docs,
            gold_section
        )

        context_parts = []

        for i, doc in enumerate(docs, start=1):
            bab_utama = doc.metadata.get("Bab Utama", "N/A")
            sub_bab = doc.metadata.get("Sub Bab", "")

            section_info = bab_utama

            if sub_bab:
                section_info += f" - {sub_bab}"

            context_parts.append(
                f"[Context {i} | {section_info}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        generation_start = time.perf_counter()

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        generation_end = time.perf_counter()

        generation_latency = generation_end - generation_start

        abstained = abstain_answer(answer)

        if answerable:
            token_f1 = token_f1_score(
                answer,
                expected_answer
            )

            rouge_l = rouge_l_f1(
                answer,
                expected_answer
            )

            keyword_recall = must_include_recall(
                answer,
                must_include
            )

            correct_abstention = None
        else:
            token_f1 = None
            rouge_l = None
            keyword_recall = None
            correct_abstention = abstained

        print(f"Gold Section: {gold_section}")
        print(f"Recall@{k}: {recall_k}")
        print(f"Evidence Recall@{k}: {evidence_recall_k}")
        print(f"First Relevant Rank: {first_rank}")
        print(f"MRR contribution: {rr}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Model Answer: {answer}")

        if answerable:
            print(f"Token F1: {token_f1:.4f}")
            print(f"ROUGE-L F1: {rouge_l:.4f}")

            if keyword_recall is not None:
                print(f"Must-Include Recall: {keyword_recall:.4f}")
        else:
            print(f"Correct Abstention: {correct_abstention}")

        result = {
            "id": question_id,
            "question": question,
            "category": category,
            "question_type": question_type,
            "difficulty": difficulty,
            "answerable": answerable,
            "K": k,
            "gold_section": gold_section,
            "gold_subsection": gold_subsection,
            "gold_evidence": gold_evidence,
            "expected_answer": expected_answer,
            "must_include": must_include,
            "retrieved_sections": [
                doc.metadata.get("Bab Utama", "N/A")
                for doc in docs
            ],
            "retrieved_contexts": [
                {
                    "rank": i,
                    "bab_utama": doc.metadata.get("Bab Utama"),
                    "sub_bab": doc.metadata.get("Sub Bab"),
                    "content": doc.page_content
                }
                for i, doc in enumerate(docs, start=1)
            ],
            "recall@k": recall_k,
            "evidence_recall@k": evidence_recall_k,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": rr,
            "model_answer": answer,
            "token_f1": token_f1,
            "rouge_l_f1": rouge_l,
            "must_include_recall": keyword_recall,
            "abstained": abstained,
            "correct_abstention": correct_abstention,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "total_latency": retrieval_latency + generation_latency
        }

        results.append(result)

    return results


def calculate_summary(results, k):
    answerable_results = [
        result for result in results
        if result["answerable"]
    ]

    unanswerable_results = [
        result for result in results
        if not result["answerable"]
    ]

    retrieval_results = [
        result for result in results
        if result["gold_section"] is not None
    ]

    summary = {
        "K": k,
        "number_of_questions": len(results),
        "retrieval": {
            "recall@k": safe_mean([
                result["recall@k"]
                for result in retrieval_results
            ]),
            "evidence_recall@k": safe_mean([
                result["evidence_recall@k"]
                for result in retrieval_results
            ]),
            "mrr": safe_mean([
                result["reciprocal_rank"]
                for result in retrieval_results
            ])
        },
        "generation": {
            "token_f1": safe_mean([
                result["token_f1"]
                for result in answerable_results
            ]),
            "rouge_l_f1": safe_mean([
                result["rouge_l_f1"]
                for result in answerable_results
            ]),
            "must_include_recall": safe_mean([
                result["must_include_recall"]
                for result in answerable_results
            ]),
            "abstention_accuracy": safe_mean([
                float(result["correct_abstention"])
                for result in unanswerable_results
            ])
        },
        "latency": {
            "average_retrieval_latency": safe_mean([
                result["retrieval_latency"]
                for result in results
            ]),
            "average_generation_latency": safe_mean([
                result["generation_latency"]
                for result in results
            ]),
            "average_total_latency": safe_mean([
                result["total_latency"]
                for result in results
            ])
        }
    }

    return summary


def run_ablation():
    summaries = []
    detailed_results = {}

    for k in K_RANGE:
        print(f"RUNNING ABLATION FOR K={k}")

        results = evaluate_with_k(
            evaluation_data,
            k
        )

        summary = calculate_summary(
            results,
            k
        )

        summaries.append(summary)

        detailed_results[str(k)] = {
            "summary": summary,
            "results": results
        }

    return summaries, detailed_results


def create_summary_dataframe(summaries):
    rows = []

    for summary in summaries:
        rows.append({
            "K": summary["K"],
            "Recall@K": summary["retrieval"]["recall@k"],
            "Evidence Recall@K": summary["retrieval"]["evidence_recall@k"],
            "MRR": summary["retrieval"]["mrr"],
            "Token F1": summary["generation"]["token_f1"],
            "ROUGE-L F1": summary["generation"]["rouge_l_f1"],
            "Must-Include Recall": summary["generation"]["must_include_recall"],
            "Abstention Accuracy": summary["generation"]["abstention_accuracy"],
            "Avg Retrieval Latency (s)": summary["latency"]["average_retrieval_latency"],
            "Avg Generation Latency (s)": summary["latency"]["average_generation_latency"],
            "Avg Total Latency (s)": summary["latency"]["average_total_latency"]
        })

    return pd.DataFrame(rows)


def save_ablation_results(df, detailed_results):
    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            detailed_results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 100)
    print(f"Summary CSV saved to: {OUTPUT_CSV}")
    print(f"Detailed JSON saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    summaries, detailed_results = run_ablation()

    df_summary = create_summary_dataframe(
        summaries
    )

    print(df_summary.to_string(index=False))

    save_ablation_results(
        df_summary,
        detailed_results
    )