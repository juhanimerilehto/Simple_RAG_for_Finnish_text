# RAG Solution GUI - Minimum Viable Product
# Version: 1.0
# Date: 2023-12-04
# Creator: Juhani Merilehto - @juhanimerilehto


import customtkinter
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import os
import textract
from transformers import AutoTokenizer, AutoModel
import numpy as np
import openai
import re
import faiss
from dotenv import load_dotenv

from RAG_full_script import extract_text, clean_text, tokenize_paragraphs, vectorize_paragraphs, query_chatgpt, build_prompt, vector_search, build_vector_index

customtkinter.set_appearance_mode("System")  # Other modes: "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Other themes: "green", "dark-blue"

class PdfProcessorApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("PDF Processor with ChatGPT Integration")
        self.geometry("1100x580")

        self.initialize_openai()
        self.initialize_model()
        self.create_widgets()

        self.vector_index = None

    def initialize_openai(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            messagebox.showerror("Error", "OpenAI API key not found.")
            self.destroy()
        openai.api_key = self.api_key

    def initialize_model(self):
        model_name = 'TurkuNLP/bert-base-finnish-uncased-v1'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def create_widgets(self):
        self.load_pdf_button = customtkinter.CTkButton(self, text="Load PDF", command=self.load_pdf)
        self.load_pdf_button.pack(pady=10, padx=10)

        self.text_area = customtkinter.CTkTextbox(self, width=500, height=150)
        self.text_area.pack(pady=10, padx=10)

        self.process_text_button = customtkinter.CTkButton(self, text="Process Text", command=self.process_text)
        self.process_text_button.pack(pady=10, padx=10)

        self.chat_response_area = customtkinter.CTkTextbox(self, width=500, height=150)
        self.chat_response_area.pack(pady=10, padx=10)

        self.query_entry = customtkinter.CTkEntry(self, width=400, height=25)
        self.query_entry.pack(pady=10, padx=10)

        self.send_query_button = customtkinter.CTkButton(self, text="Send Query", command=self.send_query)
        self.send_query_button.pack(pady=10, padx=10)

    #Visualizations here

        # Label for displaying the number of vectors created
        self.vector_count_label = customtkinter.CTkLabel(self, text="Number of vectors: 0")
        self.vector_count_label.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")

    def load_pdf(self):
        file_path = customtkinter.filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.process_text_button.configure(state="disabled")
            threading.Thread(target=self.extract_text_from_pdf, args=(file_path,)).start()

    def extract_text_from_pdf(self, file_path):
        try:
            text = textract.process(file_path).decode('utf-8')
            self.text_area.insert("end", text)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.process_text_button.configure(state="normal")

    def process_text(self):
        extracted_text = self.text_area.get("1.0", "end")
        clean = clean_text(extracted_text)
        self.paragraphs = tokenize_paragraphs(clean)
        self.paragraph_vectors = vectorize_paragraphs(self.paragraphs, self.model, self.tokenizer)
        self.vector_index = build_vector_index(self.paragraph_vectors)
        self.text_area.delete("1.0", "end")
        self.text_area.insert("end", "\n---\n".join(self.paragraphs))
        messagebox.showinfo("Info", "Text processing completed.")
        self.vector_count_label.configure(text=f"Number of vectors: {len(self.paragraph_vectors)}")

    def send_query(self):
        user_query = self.query_entry.get()
        if not user_query.strip():
            messagebox.showerror("Error", "Query cannot be empty.")
            return

        query_vector = vectorize_paragraphs([user_query], self.model, self.tokenizer)[0]
        if self.vector_index is None:
            messagebox.showerror("Error", "Text has not been processed yet.")
            return

        top_indices = vector_search(query_vector, self.vector_index)
        context = [self.paragraphs[i] for i in top_indices]

        chat_prompt = build_prompt(user_query, context)
        self.send_query_button.configure(state="disabled")
        threading.Thread(target=self.query_openai, args=(chat_prompt,)).start()

    def query_openai(self, prompt):
        try:
            response = query_chatgpt(prompt)
            self.chat_response_area.delete("1.0", "end")
            self.chat_response_area.insert("end", response)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.send_query_button.configure(state="normal")

if __name__ == "__main__":
    app = PdfProcessorApp()
    app.mainloop()

