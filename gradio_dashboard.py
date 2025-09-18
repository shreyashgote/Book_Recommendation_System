import gradio as gr
import pandas as pd
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embedding)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,

) -> pd.DataFrame:
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    book_list = [int(rec.page_content.strip().split(":")[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(book_list)].head(final_top_k)

    if category != "ALL":
        books_recs = books_recs[books_recs["simple_categories"] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Supprising":
        books_recs.sort_values(by="suprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenceful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)
    
    return books_recs

def reccommend_books(
        query: str,
        category: str,
        tone: str,
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_spit = row["authors"].split(";")
        if len(authors_spit) == 2:
            authors_str = f"{authors_spit[0]} and { authors_spit[1] }"
        elif len(authors_spit) > 2:
            authors_str = f"{', '.join(authors_spit[:-1])}, and {authors_spit[-1]}"

        else:
            authors_str = row["authors"]

        caption = f"{row["title"]} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["ALL"] + sorted(books["simple_categories"].unique())
tone = ["ALL"] + ["Happy", "Supprising", "Angry", "Suspenceful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a Description of a Book", placeholder= "e.g. , A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, value="ALL", label="Select a Category")
        tone_dropdown = gr.Dropdown(choices=tone, value="ALL", label="Select a Tone")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("### Recommendation")
    output = gr.Gallery(label = "Recommended Books", columns= 8, rows= 2)

    submit_button.click(fn = reccommend_books, inputs= [user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch()