import torch
import transformers
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import anthropic

df_aq = pd.read_excel("all_QA.xlsx")
df_aq = df_aq.apply(lambda x: x.replace('\n', ''), axis=0)
list_of_topics = sorted(df_aq['TOPIC'].unique())

# Page configuration
st.set_page_config(page_title="MaSTeA for MaScQA", page_icon=":microscope:", layout="wide")

# Seed
torch.random.manual_seed(0)

# Set up the title of the application
st.title('MaSTeA for MaScQA :microscope: :female-scientist:')
subtitle = r'''
$\textsf{
    \large \textbf{Ma}terials \textbf{S}cience \textbf{Te}aching \textbf{A}ssistant
}$
'''
st.write(f"{subtitle} ")

col1, col2 = st.columns((1,3))
col1.header("Settings")

question = ''

# Set up the title for second column
col2.header('Select your topic to study')

# Topic
topic = col2.selectbox(
   "Topics",
   options=sorted(df_aq['TOPIC'].unique()),
   index=0,  # Default value
   help="Select the material science topic"
)

question_list_matching = df_aq.loc[df_aq['TOPIC'] == topic]['QUESTION'].to_list()
question_type_matching = df_aq.loc[df_aq['TOPIC'] == topic]['Question Type'].to_list()

question_type_formated = []
for i, val in enumerate(question_type_matching):
    question_type_formated.append(f"{val}-{i+1}")
question_type_formated.insert(0, "Enter your own question")

question_selected = col2.selectbox(
   "List of question",
   options=question_type_formated,
   index=0,  # Default value
   help="Select the material science question and get the answer with steps"
)

col2.caption('MCQS - Multiple choice questions, MCQS-NUMS - Numerical MCQS, MATCH - Matching questions, NUM - Numerical questions')

if question_selected == "Enter your own question":
    question_num = 0
    
# Create a button that when clicked will output the LLMs generated output
if question_selected != "Enter your own question":
    # Select model
    model_option = col1.selectbox(
        "Select a model",
        ("ChatGPT", "GPT4", "Llama3-8b", "Phi3-mini"),
        index=0,
        placeholder="Select model",
        )
    if model_option == 'ChatGPT':
        model_answers = df_aq.loc[df_aq['TOPIC'] == topic]['detailed_chatgpt'].to_list()
    
    if model_option == 'GPT4':
        model_answers = df_aq.loc[df_aq['TOPIC'] == topic]['detailed_gpt4'].to_list()
    
    if model_option == 'Llama3-8b':
        model_answers = df_aq.loc[df_aq['TOPIC'] == topic]['detailed_llama3_8b'].to_list()

    if model_option == 'Phi3-mini':
        model_answers = df_aq.loc[df_aq['TOPIC'] == topic]['detailed_phi3_mini'].to_list()

    answer_type_matching = df_aq.loc[df_aq['TOPIC'] == topic]['Correct Answer'].to_list()
    question_num = int(question_selected.split("-")[-1])
    # Format text    
    col2.write_stream(question_list_matching[question_num - 1].removeprefix('[').removesuffix(']').replace("'"," ").replace(","," ").split("\\n"))
    if col2.button('Get answer'):
        # Format text
        col2.write_stream(model_answers[question_num - 1].removeprefix('[').removesuffix(']').replace("'"," ").replace(","," ").split("\\n"))
        col2.markdown(f'Correct answer should be {answer_type_matching[question_num - 1]}')

# Read input
if question_selected == "Enter your own question":
    # 
    system_prompt = "Solve the following question with highly detailed step by step explanation. Write the correct answer inside a dictionary at the end in the following format. The key 'answer' has a list which can be filled by all correct options or by a number as required while answering the question. For example for question with correct answer as option (a), return {'answer':[a]} at the end of solution. For question with multiple options'a,c' as answers, return {'answer':[a,c]}. And for question with numerical values as answer (say 1.33), return {'answer':[1.33]}"

    # Select token length
    token_length = col1.select_slider(
    "Token length",
    options=[256, 512, 1024, 2048],
    value=256,  # Default value
    )

    model_option = col1.selectbox("Select a model", 
                                  ("Phi-3-mini-4k-instruct", 
                                   "claude-3-haiku-20240307", 
                                   "claude-3-sonnet-20240229", 
                                   "claude-3-opus-20240229", 
                                   "claude-2.1",
                                   "claude-2.0",
                                   "claude-instant-1.2",
                                   ),
                                  index=0,
                                  placeholder="Select model")
    
    option = col2.radio("Input method",
                        ("Upload text file", "Enter text"),
                        index=1,
                        help="Choose whether to upload a text file containing your question or to enter the text manually.")
    
    if model_option != "Phi-3-mini-4k-instruct":
        text_input_container = col2.empty()
        api = text_input_container.text_input("Enter api key")

        if api != "":
            text_input_container.empty()
            api_hide = "\*"*len(api)
            col2.info(api_hide)
    
    if option == "Upload text file":
        uploaded_file = col1.file_uploader("Add text file containing question", type=["txt"])
        if uploaded_file:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            col1.text_area("Uploaded Question:", value=string_data, height=150)
            question = string_data
        else:
            col1.warning("No file uploaded!")
    elif option == "Enter text":
        question = col2.text_area("Enter your question here:")

    if model_option == "Phi-3-mini-4k-instruct":
        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            f"microsoft/{model_option}",
            device_map="cpu",
            torch_dtype="auto",
            trust_remote_code=True,)
        tokenizer = transformers.AutoTokenizer.from_pretrained(f"microsoft/{model_option}")

        # Prepare the pipeline
        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        # Set generation arguments
        generation_args = {
            "max_new_tokens": token_length,
            "return_full_text": True,
            "do_sample": True,
            "top_p": 0.9,
        }
        
        messages = [
            {"role": "user", "content": system_prompt + question},
        ]

        # Create a button that when clicked will output the LLMs generated output
        # if question_type_matching[question_num] == "Enter your own question":
        if col2.button('Generate output'):
            output = pipe(messages, **generation_args)
            model_output = output[0]['generated_text'][1]['content']
            # Display a generated output message below the button
            col2.write(f'{model_output}')
    else:
        if not api:
            col2.warning("This model requires API key")

        elif len(question) != 0:
            client = anthropic.Anthropic(api_key = api)
            message = client.messages.create(
                model=model_option,
                max_tokens=token_length,
                temperature=0,
                top_p=0.9,
                system = system_prompt, 
                messages=[{"role": "user", "content": [ 
                    {"type": "text",
                    "text": question
                    }
                    ]}])
        
            # Create a button that when clicked will output the LLMs generated output
            if col2.button('Generate output'):
                model_output = message.content
                # Display a generated output message below the button
                col2.write(f'{model_output[0].text}')