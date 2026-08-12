import json
import re
import time
import string
from collections import Counter
from statistics import mean
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

EVAL_PATH = "./evaluation_dataset.json"
OUTPUT_PATH = "./evaluation_results.json"

RECALL_K_VALUES = [1, 3, 5]
CONTEXT_K = 5

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


def evidence_recall(docs, gold_evidence):
    if not gold_evidence:
        return None

    combined_context = "\n".join(
        doc.page_content
        for doc in docs
    )

    matched = sum(
        1 for evidence in gold_evidence
        if phrase_in_text(evidence, combined_context)
    )

    return matched / len(gold_evidence)


def eval_retrieved_documents(docs, gold_section, gold_evidence):
    metrics = {}

    retrieved_sections = [
        doc.metadata.get("Bab Utama", "N/A")
        for doc in docs
    ]

    metrics["retrieved_sections"] = retrieved_sections

    for k in RECALL_K_VALUES:
        docs_at_k = docs[:k]

        if gold_section:
            recall_at_k = float(
                any(
                    document_matches_section(doc, gold_section)
                    for doc in docs_at_k
                )
            )
        else:
            recall_at_k = None

        metrics[f"recall@{k}"] = recall_at_k
        metrics[f"evidence_recall@{k}"] = evidence_recall(
            docs_at_k,
            gold_evidence
        )

    first_relevant_rank = None

    if gold_section:
        for rank, doc in enumerate(docs, start=1):
            if document_matches_section(doc, gold_section):
                first_relevant_rank = rank
                break

    if first_relevant_rank is None:
        reciprocal_rank = 0.0 if gold_section else None
    else:
        reciprocal_rank = 1.0 / first_relevant_rank

    metrics["first_relevant_rank"] = first_relevant_rank
    metrics["reciprocal_rank"] = reciprocal_rank

    return metrics


def eval_model(evaluation_data):
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
        print(f"Evaluating {idx}/{len(evaluation_data)} | ID: {question_id}")
        print(f"Question: {question}")
        print(f"Category: {category}")
        print(f"Type: {question_type}")
        print(f"Difficulty: {difficulty}")
        print(f"Answerable: {answerable}")

        retrieval_start = time.perf_counter()

        all_docs = retriever.invoke(question)

        retrieval_end = time.perf_counter()

        retrieval_latency = retrieval_end - retrieval_start

        retrieval_metrics = eval_retrieved_documents(
            all_docs,
            gold_section,
            gold_evidence
        )

        docs = all_docs[:CONTEXT_K]

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

        print(f"\nGold Section: {gold_section}")
        print(f"Gold Subsection: {gold_subsection}")
        print(f"Retrieved Sections: {retrieval_metrics['retrieved_sections']}")

        for k in RECALL_K_VALUES:
            print(
                f"Recall@{k}: "
                f"{retrieval_metrics[f'recall@{k}']}"
            )

            print(
                f"Evidence Recall@{k}: "
                f"{retrieval_metrics[f'evidence_recall@{k}']}"
            )

        print(
            f"First Relevant Rank: "
            f"{retrieval_metrics['first_relevant_rank']}"
        )

        print(
            f"Reciprocal Rank: "
            f"{retrieval_metrics['reciprocal_rank']}"
        )

        print(f"\nExpected Answer: {expected_answer}")
        print(f"Model Answer: {answer}")

        if answerable:
            print(f"Token F1: {token_f1:.4f}")
            print(f"ROUGE-L F1: {rouge_l:.4f}")

            if keyword_recall is not None:
                print(f"Must-Include Recall: {keyword_recall:.4f}")
        else:
            print(f"Abstained: {abstained}")
            print(f"Correct Abstention: {correct_abstention}")

        print(f"\nRetrieval Latency: {retrieval_latency:.4f}s")
        print(f"Generation Latency: {generation_latency:.4f}s")

        result = {
            "id": question_id,
            "question": question,
            "category": category,
            "question_type": question_type,
            "difficulty": difficulty,
            "answerable": answerable,
            "expected_answer": expected_answer,
            "must_include": must_include,
            "gold_section": gold_section,
            "gold_subsection": gold_subsection,
            "gold_evidence": gold_evidence,
            "retrieved_sections": retrieval_metrics["retrieved_sections"],
            "retrieved_contexts": [
                {
                    "rank": i,
                    "bab_utama": doc.metadata.get("Bab Utama"),
                    "sub_bab": doc.metadata.get("Sub Bab"),
                    "content": doc.page_content
                }
                for i, doc in enumerate(all_docs, start=1)
            ],
            "recall@1": retrieval_metrics["recall@1"],
            "recall@3": retrieval_metrics["recall@3"],
            "recall@5": retrieval_metrics["recall@5"],
            "evidence_recall@1": retrieval_metrics["evidence_recall@1"],
            "evidence_recall@3": retrieval_metrics["evidence_recall@3"],
            "evidence_recall@5": retrieval_metrics["evidence_recall@5"],
            "first_relevant_rank": retrieval_metrics["first_relevant_rank"],
            "reciprocal_rank": retrieval_metrics["reciprocal_rank"],
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


def safe_mean(values):
    values = [
        value for value in values
        if value is not None
    ]

    if not values:
        return None

    return mean(values)


def calculate_metrics(results):
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

    retrieval_summary = {}

    for k in RECALL_K_VALUES:
        retrieval_summary[f"recall@{k}"] = safe_mean([
            result[f"recall@{k}"]
            for result in retrieval_results
        ])

        retrieval_summary[f"evidence_recall@{k}"] = safe_mean([
            result[f"evidence_recall@{k}"]
            for result in retrieval_results
        ])

    retrieval_summary["mrr"] = safe_mean([
        result["reciprocal_rank"]
        for result in retrieval_results
    ])

    generation_summary = {
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
        ])
    }

    if unanswerable_results:
        generation_summary["abstention_accuracy"] = safe_mean([
            float(result["correct_abstention"])
            for result in unanswerable_results
        ])
    else:
        generation_summary["abstention_accuracy"] = None

    latency_summary = {
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

    category_summary = calculate_group_metrics(
        results,
        group_key="category"
    )

    question_type_summary = calculate_group_metrics(
        results,
        group_key="question_type"
    )

    difficulty_summary = calculate_group_metrics(
        results,
        group_key="difficulty"
    )

    summary = {
        "number_of_questions": len(results),
        "number_of_answerable_questions": len(answerable_results),
        "number_of_unanswerable_questions": len(unanswerable_results),
        "retrieval": retrieval_summary,
        "generation": generation_summary,
        "latency": latency_summary,
        "by_category": category_summary,
        "by_question_type": question_type_summary,
        "by_difficulty": difficulty_summary
    }

    return summary


def calculate_group_metrics(results, group_key):
    groups = {}

    group_names = sorted(
        set(
            result[group_key]
            for result in results
        )
    )

    for group_name in group_names:
        group_results = [
            result for result in results
            if result[group_key] == group_name
        ]

        answerable_results = [
            result for result in group_results
            if result["answerable"]
        ]

        retrieval_results = [
            result for result in group_results
            if result["gold_section"] is not None
        ]

        group_metrics = {
            "count": len(group_results)
        }

        for k in RECALL_K_VALUES:
            group_metrics[f"recall@{k}"] = safe_mean([
                result[f"recall@{k}"]
                for result in retrieval_results
            ])

        group_metrics["mrr"] = safe_mean([
            result["reciprocal_rank"]
            for result in retrieval_results
        ])

        group_metrics["token_f1"] = safe_mean([
            result["token_f1"]
            for result in answerable_results
        ])

        group_metrics["rouge_l_f1"] = safe_mean([
            result["rouge_l_f1"]
            for result in answerable_results
        ])

        group_metrics["must_include_recall"] = safe_mean([
            result["must_include_recall"]
            for result in answerable_results
        ])

        groups[group_name] = group_metrics

    return groups


def save_results_to_json(results, summary, output_path):
    output_data = {
        "configuration": {
            "model": "qwen2.5:3b",
            "temperature": 0.0,
            "context_k": CONTEXT_K,
            "recall_k_values": RECALL_K_VALUES
        },
        "summary": summary,
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    results = eval_model(evaluation_data)
    summary = calculate_metrics(results)

    print(json.dumps(
        summary,
        ensure_ascii=False,
        indent=4
    ))

    save_results_to_json(
        results,
        summary,
        OUTPUT_PATH
    )