from chatbot import get_executor, generate_response
from dotenv import load_dotenv

load_dotenv()

# Different prompts used to test the chatbot

# prompt = """Ein Klosterschüler im Mittelalter durchläuft nach seiner Aufnahme in die Schule
# einen Kurs, der sich am Konzept der Septem Artes orientiert. Den Grundkurs hat
# er bereits erfolgreich absolviert.

# Welches der nachfolgend genannten Fächer muss er im Aufbaukurs absolvieren?"""

# prompt = "Was ist die Zusammenfassung von Seminar 3?"

# prompt = "Wann findet der Klausur statt?"

# prompt = "Wie genau werden die Schreibaufgaben auf die Klausurnote angerechnet?"

# prompt = "Was ist die Zusammenfassung von Vorlesung 3 und Vorlesung 1? War auch Fend Biologe? After getting the answer then make it in bullet points."

# prompt = "Was ist die Zusammenfassung von Vorlesung 3 und Vorlesung 1? War auch Fend Biologe?"

# prompt = "Was ist die Zusammenfassung von Vorlesung 3 und Vorlesung 1? War auch Fend Biologe? Formulate your answers as bullet points."

# prompt = "Was ist die Zusammenfassung von Vorlesung 3 und Vorlesung 1? die Zusammenfassung von Seminar 3?"

# prompt = "Was ist die Zusammenfassung von Vorlesung 3?"

prompt = "War Fend Biologe?"

agent_executor, conversational_memory = get_executor("All Material")

response, explanation, openai_callback = generate_response(prompt, agent_executor, conversational_memory, None)

print("Response: ", response["output"])

print(openai_callback)


# prompt = "Was ist meine letzte Frage?"

prompt = "Was hat er für die Bildung gemacht?"

response, explanation, openai_callback = generate_response(prompt, agent_executor, conversational_memory, None)

print("Response: ", response["output"])

print(openai_callback)

