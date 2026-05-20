from youtube_transcript_api import YouTubeTranscriptApi
import ollama
import re

def extract_video_api(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("URL YouTube invalide")
    return match.group(1)

def get_transcript(video_url):

    video_id = extract_video_api(video_url)

    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(
        video_id,
        languages=["fr", "en"]
    )

    text = " ".join([item.text for item in transcript])

    return text

def summarize_and_translate(text):
    prompt= f"""Tu es un agent IA qui a pour but de resumer la retranscription d'une vidéo youtube
        {text}
        
        Tache:
        1.Traduire le contenu
        2.Résumer clairement la vidéo
        3.Données les idées globale
        4.Faire une conclusion

        Réponds en FRANÇAIS
    """
    response = ollama.chat(
        model="qwen3.5:4b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def youtube_agent(video_url):
    transcript = get_transcript(video_url)
    result = summarize_and_translate(transcript)
    return result

def start():
    url="https://www.youtube.com/watch?v=z3NpVq-y6jU"
    print(youtube_agent(url))
