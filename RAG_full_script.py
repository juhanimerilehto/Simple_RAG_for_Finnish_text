# RAG Solution for Document Processing
# Version: 1.0
# Date: 2023-12-04
# Creator: Juhani Merilehto - @juhanimerilehto

# Import dependencies
import os
import re
from dotenv import load_dotenv
from timeit import default_timer as timer
import numpy as np
import textract
from openai import OpenAI
import faiss  # Vector similarity index
from transformers import AutoTokenizer, AutoModel

load_dotenv()

# Initialize OpenAI client and configure system
client = OpenAI()
pdf_dir = "./data/"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
max_tokens = 4096
index_vecs = 768  # Adjust this to match the output size of your model / Säädä tämä vastaamaan mallisi tulosteen kokoa

# Load Finnish BERT model / Lataa suomalainen BERT-malli
model_name = 'TurkuNLP/bert-base-finnish-uncased-v1'  # Example model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# PDF Processing Pipeline / PDF:n käsittelyn työvaiheet
def extract_text(pdf_path):
    # Extract text from pdf / Poimi teksti pdf:stä
    text = textract.process(pdf_path)
    print(f"Extracted text from {pdf_path}:\n{text[:25]}...")  # Print the first 25 characters / printtaa ekat 25 merkkiä testiksi
    return text.decode("utf-8") if isinstance(text, bytes) else text

def clean_text(text):
    # Clean extracted text / Puhdista poimittu teksti
    # Removes special characters, preserving letters, punctuation, and whitespace
    # Poistaa erikoismerkit, säilyttäen kirjaimet, välimerkit ja välilyönnit
    cleaned_text = re.sub(r'[^\w\s\u00C0-\u017F\u0400-\u04FF.,!?;:"\'-]', '', text)
    return cleaned_text

def is_bullet_point(paragraph):
    # Identify if a paragraph is a bullet point / Tunnista, onko kappale luettelomerkki
    bullet_point_pattern = r'^\s*(\*|\-|•|\d+\.\s|\w+\)\s)'
    return bool(re.match(bullet_point_pattern, paragraph.strip()))

def split_paragraph(paragraph, max_length):
    # Split a paragraph into smaller chunks with a maximum length
    # Jaa kappale pienempiin osiin enimmäispituuden mukaan
    words = paragraph.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i+max_length])

def tokenize_paragraphs(text, min_word_count=30, max_length=512):
    # Split text into paragraphs / Jaa teksti kappaleiksi
    paragraphs = text.split('\r\n\r\n')  # Splitting by two newlines / Jaetaan kahdella rivinvaihdolla
    processed_paragraphs = []
    previous_paragraph = ""

    for paragraph in paragraphs:
        words = paragraph.split()
        
        # Check for long paragraphs and split them / Tarkista pitkät kappaleet ja jaa ne
        if len(words) > max_length:
            for chunk in split_paragraph(paragraph, max_length):
                if len(chunk.split()) >= min_word_count:
                    processed_paragraphs.append(chunk)
            continue

        if previous_paragraph and paragraph and is_bullet_point(paragraph):
            if previous_paragraph.strip().endswith(':'):
                processed_paragraphs[-1] += ' ' + paragraph
                continue

        if len(words) >= min_word_count or is_bullet_point(paragraph):
            processed_paragraphs.append(paragraph)
            previous_paragraph = paragraph
        else:
            previous_paragraph = ""

    print(f"Number of paragraphs extracted: {len(paragraphs)}")  # Print how many paragraphs / Tulosta kuinka monta kappaletta
    # Save processed paragraphs to a text file / Tallenna käsitellyt kappaleet tekstitiedostoon
    with open('./processed_paragraphs.txt', 'w', encoding='utf-8') as file:
        for i, paragraph in enumerate(processed_paragraphs):
            file.write(f"Paragraph {i + 1}:\n{paragraph}\n")
            file.write("\n---\n")  # Delimiter between paragraphs / Erotin kappaleiden välillä

    print("Processed paragraphs saved to './processed_paragraphs.txt'")
    print(f"Number of processed paragraphs: {len(processed_paragraphs)}")
    return processed_paragraphs

def vectorize_paragraphs(paragraphs, model, tokenizer):
    # Vectorize paragraphs using a given model and tokenizer / Vektoroi kappaleet käyttämällä annettua mallia ja tokenizeria
    vectors = []
    for p in paragraphs:
        inputs = tokenizer(p, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        vectors.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    if not vectors:
        print("Warning: No vectors were created. The paragraphs list might be empty or invalid.")
        return np.array([])
    return np.vstack(vectors)

# ChatGPT Integration
def query_chatgpt(prompt):
    # Query ChatGPT model with a given prompt using OpenAI's updated API / Tee kysely ChatGPT-malliin annetulla kehotteella käyttäen OpenAI:n päivitettyä API:a
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # Using the latest preview version / Käyttäen uusinta esikatseluversiota GPT-4-1106-preview
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    # Extracting the response content / Vastauksen sisällön poiminta
    if response.choices:
        # Get the content of the last message in the chat, which should be the model's response
        # Hae keskustelun viimeisen viestin sisältö, joka pitäisi olla mallin vastaus
        return response.choices[0].message.content
    else:
        return "No response received."

def build_prompt(question, context):
    # Construct a prompt for ChatGPT from question and context paragraphs
    # Muodosta kehote ChatGPT:lle kysymyksestä ja kontekstikappaleista
    prompt = f'Olet Keski-Suomen hyvinvointialueen tekoälyassistentti. Käytössäsi on joitakin hyvinvointialueen dokumentteja, joista saat olennaista sisältöä mitä käyttää vastauksessasi. Sisällöt:\n\n'
    prompt += '\n\n'.join([f'Context {i+1}: {item}' for i, item in enumerate(context)])
    prompt += f'\n\nQuestion: {question}\nResponse:'
    print(f"Prompt for GPT-4:\n{prompt[:4500]}...")  # Print the first 4500 characters of the prompt / Tulosta kehotteen ensimmäiset 4500 merkkiä nopeaan kontekstintarkastukseen
    return prompt

# Local Vector Search Index
# Paikallinen vektorihakemisto
def build_vector_index(vectors):
    # Build an approximate nearest neighbors index for fast search / Rakenna likimääräinen lähimpien naapurien indeksi nopeaa hakua varten
    index = faiss.IndexFlatIP(index_vecs)
    index.add(vectors)
    return index

def vector_search(query_vector, index, top_k=10):
    # Find top 5 most similar vectors in the index / Etsi viisi samankaltaisinta vektoria indeksistä
    D, I = index.search(np.array([query_vector]), top_k)
    return I[0]

# Execution Orchestration
# Suorituksen orkestrointi
if __name__ == "__main__":
    start = timer()

    # Load PDFs and extract text / Lataa PDF:t ja poimi teksti
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    paragraphs = []
    for p in pdfs:
        path = os.path.join(pdf_dir, p)
        text = extract_text(path)
        clean = clean_text(text)
        paragraphs.extend(tokenize_paragraphs(clean))

#lets print out the paragraphs into a file for inspection / Printataan kappaleet tiedostoon tarkasteltavaksi
    output_file_path = './extracted_paragraphs.txt'
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for i, paragraph in enumerate(paragraphs):
            file.write(f"Paragraph {i + 1}:\n{paragraph}\n")
            file.write("\n---\n")  # Delimiter between paragraphs / Jako kappaleiden väliin

    print(f"Extracted paragraphs written to {output_file_path}")

    # Vectorize paragraphs / Kappaleet vektoreiksi
    vectors = vectorize_paragraphs(paragraphs, model, tokenizer)
    if vectors.size > 0:
        print(f"Total vectors created: {vectors.shape[0]}")
        print(f"Shape of vectors array: {vectors.shape}")
    else:
        print("No vectors were created.")

    vector_index = build_vector_index(vectors)

    # User interaction loop / Käyttäjän vuorovaikutussilmukka
    while True:
        question = input("Enter question (q to quit): ")
        if question.lower() == 'q':
            break

        query_vector = vectorize_paragraphs([question], model, tokenizer)[0]
        top_indices = vector_search(query_vector, vector_index)
        context = [paragraphs[i] for i in top_indices]

        chat_prompt = build_prompt(question, context)
        response = query_chatgpt(chat_prompt)
        print(response)
    
    print(f"System completed in {timer() - start:.4f} secs")

