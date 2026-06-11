from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

llm= ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

template= """
Summarize the following article in 1 consise sentence
{article}
"""

prompt= PromptTemplate.from_template(template)
parser= StrOutputParser()

chain= prompt | llm | parser

long_ahh_article="The patient is a 45-year-old male who presented with a 3-day history of intermittent fever, productive cough, mild shortness of breath, and generalized fatigue, with symptoms worsening on exertion and no reported chest pain, hemoptysis, or recent travel history. Vital signs on admission showed a temperature of 38.4 C, heart rate of 102 beats per minute, blood pressure of 128/82 mmHg, respiratory rate of 22 breaths per minute, and oxygen saturation of 95% on room air. Physical examination revealed bilateral coarse breath sounds over the lower lung fields without wheezing, and mild pharyngeal congestion was also noted. Laboratory investigations demonstrated a slightly elevated white blood cell count with neutrophil predominance, while chest radiography showed mild patchy infiltrates in the right lower lobe consistent with an early infectious process. The patient was started on empiric oral antibiotics, antipyretics, hydration, and rest, with advice for close outpatient follow-up and immediate reassessment if symptoms worsen or if oxygen saturation declines."

summary= chain.invoke({"article":long_ahh_article})

print(summary)