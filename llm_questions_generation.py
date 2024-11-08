import anthropic
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()

client = anthropic.Anthropic()

message = """
You are  IELTS instructor. Your task is to generate 5 questions for the reading section for the given text.
Types of questions:
- Multiple choice
- True/False/Not Given
- Short answer question

Read the text below in Arabic and generate questions in Arabic.

{}
"""


if __name__ == "__main__":
  
    df = pd.read_csv("subsampling_qna_enriched_with_theme_and_toxicity.tsv", sep="\t")
    
    df = df[(df["theme"].str.contains("Culture")) | (df["theme"].str.contains("History"))]
    
    df = df.sample(n=5)
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        
        text = row["text"]
        text = text.replace("\n", " ")
        # Request a chat response from the assistant
        response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": message.format(text) 
            }
        ]
        )
        print(text)
        print(response.content[0].text)
