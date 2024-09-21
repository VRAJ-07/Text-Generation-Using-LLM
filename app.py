from flask import Flask, request, jsonify, session
from langchain.prompts import PromptTemplate
import os
import torch
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from transformers.pipelines import  pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA


import json
# Instantiate Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'KEY'
# Define the template for the prompt
prompt_template_twitter = """
Generate a Twitter post sequence based on the following details:

Historical Context: {history}
Most recent input: {input}

Requirements:
- Write the tweet in English.
- Ensure the content is engaging and concise, suitable for Twitter’s character limit.
- Use emojis to enhance the expression where appropriate.
- Include relevant hashtags to increase visibility and engagement.

Note: Keep the tone light and fun, and make sure the tweet is relevant to the current topic.
- Always provide answers based on the historical context whenever it is required to address the question effectively.
"""
prompt_template_linkedin = """
Generate a professional LinkedIn post based on the following details:

Historical Context: {history}

Post topic provided by user:
Human: {input}

Requirements:
- The language should be professional yet inviting to attract industry professionals, researchers, and enthusiasts.
- Convey excitement about the event or topic to spark interest and engagement.
- Highlight key topics that are relevant and thought-provoking to encourage discussions among the LinkedIn community.
- Use emojis sparingly to add a touch of enthusiasm without compromising the professional tone.
- Include relevant hashtags to enhance visibility and networking potential if appropriate.

Note: Ensure the post is succinct and impactful, designed to stimulate interaction and professional dialogue.
- Always provide answers based on the historical context whenever it is required to address the question effectively.
"""
prompt_template_proposal = """
Generate a detailed proposal based on the following specifications:

Historical Context: {history}

Specific request or need outlined by the user:
Human: {input}

Requirements:
- The proposal should be clear and well-structured, detailing the objectives, methods, expected outcomes, and benefits of the proposed idea or project.
- Use formal and persuasive language to convincingly present the proposal to stakeholders or decision-makers.
- Include key data points or evidence that support the feasibility and potential impact of the proposal.
- Emphasize how the proposal aligns with the goals or interests of the audience, highlighting any unique advantages or opportunities it presents.

Note: Ensure that the proposal is comprehensive yet concise, making it easy for the reader to understand the key messages and make informed decisions.
- Always provide answers based on the historical context whenever it is required to address the question effectively.
"""
prompt_template_question_answering = """
Generate a response to the following query based on the provided context and the specific question asked by the user:

Historical Context: {history}

User's query:
Human: {input}

Requirements:
- Provide a clear and concise answer that directly addresses the user's question.
- Include relevant facts or explanations to ensure the response is informative.
- Use a neutral and professional tone suitable for a broad audience.
- If the question involves complex topics or concepts, simplify the explanation without losing essential details.
- Whenever applicable, offer additional resources or directions for further exploration of the topic.

Note: Aim to educate and clarify, ensuring the user gains a deeper understanding of the questions.
- Always provide answers based on the historical context whenever it is required to address the question effectively.
"""
def run_model(model_name,temp,top_p):
     
    # model_path = os.path.join(f"/app/models/{model_name}")
    model_path=(r"\models\Mistral-7B-Instruct-v0.2")
    # quantization_config = BitsAndBytesConfig(
    #     load_in_16bit=True,
    #     bnb_16bit_compute_dtype=torch.float16,
    #     bnb_16bit_quant_type="nf4",
    #     bnb_16bit_use_double_quant=True,
    # )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
        quantization_config=quantization_config
    )
    # top_p means diversity of the generated text
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.max_new_tokens = 4096
    generation_config.temperature = temp
    generation_config.top_p = top_p
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
       
    )
    llm = HuggingFacePipeline(
        pipeline=llm_pipeline,
    )
    return llm
def create_vector_db(data):
    splitter = RecursiveJsonSplitter(max_chunk_size=1000)
    texts = splitter.split_text(json_data=data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device':'cpu'})
    db = FAISS.from_texts(texts, embeddings)
    print('vector db run successfully')
    return db


def retrieval_qachain(llm,question, History):
    
        # Create a vector database from the historical context
        vector_db = create_vector_db(History)
        
        # Define the prompt template
        prompt_template = """
        # History Retrieval Prompt
        - Use the provided historical context to retrieve relevant information.
        - Your task is to retrieve similar information from the historical context provided.
        - Only provide answers based on the information available in the historical context.
        - If you don't know the answer, respond with 'I don't know.' Do not merge different document data or don't make answer
        - write the answer of the question after a keyword that is 'Retrieval_Answer'
  
        {context}
        
        Question: {question}
        
        """
        
        # Create a PromptTemplate object
        prompt = PromptTemplate(input_variables=['question','context'], template=prompt_template)
        
        # Create a RetrievalQA object
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,  # Replace 'llm' with your model
            # chain_type="stuff", 
            retriever=vector_db.as_retriever(), 
            chain_type_kwargs={"prompt": prompt},
            verbose=True
        )
        
        # Run the QA chain to retrieve historical information

        History = qa_chain(question)
        print(History)
        History=str(History)
        index = History.find("Retrieval_Answer")
        second_index=History.find("Retrieval_Answer",index+len("Retrieval_Answer"))
        if index == -1:
             return "The word 'Answer' was not found in the string."
        start_index = second_index + len("Retrieval_Answer:")
        print(start_index)
        result = History[start_index:].strip()  
        
        print(result)    
        # print(History)
        return result
   

# Define the Flask route
@app.route('/conversation', methods=['POST'])
def converse():
    
        # user_input = request.json.get('input')
        # prompt_type = request.json.get('prompt_type')  
        # history = session.get('history', [])
        # history.append(user_input)
        # session['history'] = history
        # session['input'] = user_input
        data=request.json
        # print(data)
        
        
        # print('history is ',History)
        user_input=data.get('prompt')
        model_name=data.get("model_name")
        prompt_type=data.get('contentfor')
        temp=data.get('temp',0.5)
        top_p=data.get('top_p',0.5)
        
        keys_to_ignore = ["Input", "Prompt_Type","model_name","contentfor","temp","top_p"]
        History= {key: value for key, value in data.items() if key not in keys_to_ignore}
        # print("first history",History)
        # History=json.dumps(History,indent=4)
        
        # print(prompt_type)
        llm=run_model(model_name,temp,top_p)
        History=retrieval_qachain(llm,user_input,History)
        print('retrieval qa chain isnow over')
        # Select the appropriate prompt based on user input
        if prompt_type == 'linkedin':
            prompt = PromptTemplate(input_variables=['history', 'input'], template=prompt_template_linkedin)
        elif prompt_type=='twitter':
            prompt = PromptTemplate(input_variables=['history', 'input'], template=prompt_template_twitter)
        elif prompt_type=='proposal generation':
            prompt = PromptTemplate(input_variables=['history', 'input'], template=prompt_template_proposal)
        elif prompt_type=='question answering':
            prompt = PromptTemplate(input_variables=['history','input'],template=prompt_template_question_answering)
     
        print('history over')
        conversation=LLMChain(
            llm = llm,
            prompt=prompt,
            verbose=True
        )
     
        llm_response = conversation.run(history=History, input=user_input)

        return jsonify({'response': llm_response})   
# Run the Flask app
if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5010, debug=True)
