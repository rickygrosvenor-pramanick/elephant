import pandas as pd
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from dotenv import load_dotenv
from alive_progress import alive_bar

# RUN python generate_advice.py 100 in the philosophic-coach directory

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-pro")

INPUT_FILE = "data/final_posts.csv"
OUTPUT_FILE = "data/final_posts_with_advice.csv"
BOOK_FILE = "data/nicomachean-ethics"

with open(BOOK_FILE, 'r', encoding="utf-8") as f:
    book_text = f.read()

df = pd.read_csv(INPUT_FILE)
df = df[['post_id', 'post_title', 'post_body']]  

num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
df = df.head(num_samples)

if os.path.exists(OUTPUT_FILE):
    df_out = pd.read_csv(OUTPUT_FILE)
else:
    df_out = df.copy()
    df_out['advice_no_book'] = ""
    df_out['advice_with_book'] = ""

chat_session = model.start_chat(history=[
    {"role": "user", "parts": [
        "You are an advice assistant. Use Aristotle's *Nicomachean Ethics* as a reference.",
        f"Here is the full text:\n\n{book_text}"
    ]}
])

def process_post(i, post_text):
    try:
        no_book_prompt = f"""
        Provide helpful, concise advice for this Reddit post.
        Return ONLY the advice text (3-5 sentences). No intros or extra text.
        Post: {post_text}
        """
        no_book_resp = model.generate_content(no_book_prompt).text.strip()

        with_book_prompt = f"""
        Provide advice using Aristotle's *Nicomachean Ethics*.
        Return ONLY the advice text (3-5 sentences). No intros or extra text.
        Post: {post_text}
        """
        with_book_resp = chat_session.send_message(with_book_prompt).text.strip()

        return i, no_book_resp, with_book_resp
    except Exception as e:
        return i, f"ERROR: {e}", f"ERROR: {e}"

max_workers = 10
futures = []

with ThreadPoolExecutor(max_workers=max_workers) as executor, alive_bar(len(df_out), title="Generating Advice...", bar="classic") as bar:
    for i in range(len(df_out)):
        if pd.isna(df_out.loc[i, 'advice_no_book']) or df_out.loc[i, 'advice_no_book'] == "":
            futures.append(executor.submit(process_post, i, df_out.loc[i, 'post_body']))

    for idx, future in enumerate(as_completed(futures), 1):
        i, no_book, with_book = future.result()
        df_out.loc[i, 'advice_no_book'] = no_book
        df_out.loc[i, 'advice_with_book'] = with_book

        if idx % 10 == 0:
            df_out.to_csv(OUTPUT_FILE, index=False)

        bar()  

df_out.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Completed! Saved to {OUTPUT_FILE}")
