import pandas as pd
import json
from chatbot import get_executor, generate_response
import langsmith
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client
from langsmith.utils import LangSmithError
from dotenv import load_dotenv

load_dotenv()

def extract_lecture_questions_answers(file_path, sheet_name, output_path):

    # Read the first sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Create the first list of dictionaries
    questions_answers = [{'query': row['Frage'], 'answer': row['Zusammenfassung/Antwort']} for index, row in df.iterrows()]

    # Save the lists to JSON files
    with open(output_path, 'w') as f:
        json.dump(questions_answers, f)


def extract_seminar_questions_answers(file_path, sheet_name, output_path):

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Create the first list of dictionaries for the third sheet
    questions_answers = [{'query': row['Frage'], 'answer': row['Antwort']} for index, row in df.iterrows()]

    # Save the lists to JSON files
    with open(output_path, 'w') as f:
        json.dump(questions_answers, f)


def extract_organisational_questions_answers_v8(file_path, sheet_name, output_path):

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Filter the DataFrame to include only rows with non-empty 'Quelle Antwort Sommersemester (NEU)'
    df_filtered = df[df['Quelle Antwort Sommersemester (NEU)'].notnull() & (df['Quelle Antwort Sommersemester (NEU)'] != '')]
    # df_filtered = df_filtered[df_filtered['Quelle Antwort Sommersemester (NEU)'].isnull() | (df_filtered['Quelle Antwort Sommersemester (NEU)'] == '')]

    # Create the first list of dictionaries for the third sheet
    questions_answers = [{'query': row['Frage'], 'answer': row['Antwort']} for index, row in df_filtered.iterrows()]

    # Save the lists to JSON files
    with open(output_path, 'w') as f:
        json.dump(questions_answers, f)


def extract_organisational_questions_answers_v7(file_path, sheet_name, output_path):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Filter the DataFrame to include only rows with empty 'Quelle Antwort Sommersemester (NEU)'
    df_filtered = df[df['Quelle Antwort Sommersemester (NEU)'].isnull() | (df['Quelle Antwort Sommersemester (NEU)'] == '')]

    # Create the first list of dictionaries for the third sheet
    questions_answers = [{'query': row['Frage'], 'answer': row['Antwort']} for index, row in df_filtered.iterrows()]

    # Save the lists to JSON files
    with open(output_path, 'w') as f:
        json.dump(questions_answers, f)


def get_questions_answers(file_path):
    # Open the JSON file and load its content
    with open(file_path, 'r') as file:
        questions_answers = json.load(file)

    return questions_answers


def evaluate_sample(file_path, sample_number):

    query_answer_list = get_questions_answers(file_path)

    query_answer_sample_list = [query_answer_list[sample_number]]

    prompt = query_answer_sample_list[0]["query"]

    agent_executor, conversational_memory = get_executor("All Material")

    response, explanation, openai_callback = generate_response(prompt, agent_executor, conversational_memory)

    prediction_list = [{"result": response["output"]}]

    llm = ChatOpenAI(model="gpt-4", temperature=0, verbose=False)

    eval_chain = QAEvalChain.from_llm(llm)

    graded_outputs = eval_chain.evaluate(examples=query_answer_sample_list, predictions=prediction_list)

    print(f"Example {sample_number}:")
    print("Question: " + query_answer_sample_list[0]["query"])
    print("Real Answer: " + query_answer_sample_list[0]["answer"])
    print("Predicted Answer: " + prediction_list[0]["result"])
    print("Predicted Grade: " + graded_outputs[0]["results"])
    print()


# This is what we will evaluate
def formulate_input(question):

    return {'input': question["input"], 'chat_history': []}


# This is what we will evaluate
def formulate_output(response):
    return response["output"]


def evaluate_dataset(file_path, dataset_name):
    query_answer_list = get_questions_answers(file_path)

    #query_answer_list = [query_answer_list[2]]

    query_answer_list = query_answer_list[0:20]

    agent_executor, conversational_memory = get_executor("All Material")

    chain = formulate_input | agent_executor | formulate_output

    client = Client()

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print("Using existing dataset: ", dataset.name)
    except LangSmithError:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="",
        )
        for question_answer_pair in query_answer_list:
            client.create_example(
                inputs={"input": question_answer_pair["query"]},
                outputs={"answer": question_answer_pair["answer"]},
                dataset_id=dataset.id,
            )

        print("Created a new dataset: ", dataset.name)

    llm = ChatOpenAI(model="gpt-4", temperature=0, verbose=False)

    evaluation_config = RunEvalConfig(
        evaluators=[
            "qa",
        ],
        eval_llm=llm,
        input_key="input"
    )

    client = Client()
    run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=lambda: chain,
        client=client,
        evaluation=evaluation_config,
        verbose=True,
        concurrency_level=2
    )



def export_test_results(test_name, output_path):
    client = langsmith.Client()

    df = client.get_test_results(project_name=test_name)
    df.to_excel(output_path, index=False)



#evaluate_sample("evaluation/data/lecture_questions_answers.json", 1)

#evaluate_dataset("evaluation/data/seminar_questions_answers.json", "dataset-llm-qa-chatbot-seminar-new-v2-sample-0-10")

evaluate_dataset("evaluation/data/organisational_questions_answers_information_v7.json", "dataset-llm-qa-chatbot-lecture-new-v2")

#evaluate_dataset("evaluation/data/lecture_questions_answers.json", "dataset-llm-qa-chatbot-organizational-new-v2-sample-2")

test_name = "slight-scale-48"  
output_path = "/Users/hiteshkotte/Documents/DFKI/LLM Project/biwi-qa-system-main/LLM/evaluation/results/lecture_evaluation_results.xlsx"  
export_test_results(test_name, output_path)


