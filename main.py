import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama




load_dotenv()




def main():
    print("Hello from langchain-course!")
    information = """
Elon Reeve Musk (pronounced / ˈ i : l ɒ n ˈ m ʌ s k /; born 28 June 1971 in Pretoria [ 4 ] ) is a South African entrepreneur, founder, co-founder, or financier of SpaceX , Tesla , Neuralink , X.com (part of PayPal ), The Boring Company [ 5 ] , and xAI . He is a South African who lives and works in the United States (he holds South African, Canadian, and American citizenship). He is the CEO and CTO of SpaceX and the CEO and chief architect of Tesla Inc. In January 2021, he was named the world's richest person by Forbes magazine and Bloomberg . [ 6 ] [ 7 ] [ 8 ] Since October 28, 2022, he has owned X (formerly "Twitter"). From January 20 to May 28, 2025, he headed the Department of Government Efficiency ( DOGE ) in Donald Trump's second cabinet . As of December 21 , 2025, according to Forbes magazine , his net worth is estimated at 749 billion United States dollars (USD) [ 9 ] . ﻿[in other languages]﻿[in other languages]

"""

    summary_template = """
    given the information {information} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template = summary_template
    )


    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",  # stable, supported
        temperature=0
)

    # llm = ChatOllama(temperature=0, model="gpt-oss:20b")


    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
