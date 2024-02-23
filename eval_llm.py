from sklearn.metrics import accuracy_score

# Generate answers for a set of questions
questions = [
    "Who is the Global Chief Executive Officer of Mastek?",
    "What was the revenue for Q1FY23?",
    "What was the revenue from operations difference between Q1FY23 and Q4FY22?",
    "What was the net profit difference between Q1FY23 and Q1FY22?",
    "What was the operating EBITDA margin for the full year FY23?",
]
reference_answers = [
    "The Global Chief Executive Officer of Mastek is Hiral Chandrana.",
    "The revenue for Q1FY23 is 709.2 crore.",
    "The revenue from operations in Q1FY23 was Rs 709.2 crore, and in Q4FY22 it was Rs 581.5 crore. Therefore, the revenue from operations difference between Q1FY23 and Q4FY22 was Rs 127.7 crore.",
    "The net profit for Q1FY23 was Rs 170.6 crore, and for Q1FY22, it was Rs 161.7 crore. The net profit difference between Q1FY23 and Q1FY22 was Rs 8.9 crore.",
]
llm_answers = [
    [
        "The Global Chief Executive Officer of Mastek is Hiral Chandrana.",
        "I don't have that information.",
        "The net profit for Q1FY23 was Rs 170.6 crore, and for Q1FY22 it was Rs 161.7 crore. The net profit difference between Q1FY23 and Q1FY22 was Rs 8.9 crore.",
        "The net profit for Q1FY23 was Rs 170.6 crore, and for Q1FY22 it was Rs 161.7 crore. The difference in net profit between Q1FY23 and Q1FY22 was Rs 8.9 crore.",
    ],
    [
        "The current CEO of Mastek is John Owen.",
        "I'm sorry, but I don't have access to real-time data, and my training only goes up until January 2022. Therefore, I cannot provide you with the revenue for Q1FY23",
        "The revenue from operations in Q1FY23 was Rs 709.2 crore, and in Q4FY22 it was Rs 581.5 crore. Therefore, the difference in revenue from operations between Q1FY23 and Q4FY22 was Rs 127.7 crore.",
        "as",
    ],
    ["4", "5", "6", "as"],
]

exact_match_scores = []
for llm_answers_for_question in llm_answers:
    exact_match_scores.append(
        [
            1 if answer == reference else 0
            for answer, reference in zip(llm_answers_for_question, reference_answers)
        ]
    )

average_exact_match_scores = [
    sum(scores) / len(scores) for scores in zip(*exact_match_scores)
]
llm_names = ["gpt-neo-2.7B", "gpt-3.5-turbo", "test-model-name"]

for llm_name, score in zip(llm_names, average_exact_match_scores):
    print(f"{llm_name}: {score}")


overall_exact_match_score = accuracy_score(reference_answers, llm_answers[0])
print(f"Overall Exact Match Score: {overall_exact_match_score}")
