# Agent Ai

## Nécessaire

Pour le bon fonctionnement de l'agent AI il est nécessaire d'avoir ollama d'installer et de lancer avec le modèle qwen3.5:4b. Si vous avez pas ce modèle vous pouvez le modifier dans la fonction summarize_and_translate d'ai_agent.py.

Pour faire tourner le code vous avez besoin des librairies ollama et YouTubeTranscriptApi

```python
pip install ollama
pip install YouTubeTranscriptApi
```

## Fonctionnement

Le programme se lance avec le main.py qui fait appel a start() ai_agent.py.

La vidéo youtube a résumer se trouve dans la variable url qui est modifiable.

Tout d'abord le code va vérifier que le lien est bien un lien youtube grace a la fonction ` extract_video_api(url)` si le lien est bon la fonction retourne le lien sinon affiche une erreur.

Ensuite la fonction ` get_transcript(video_url)` va récuperer les sous-titres anglais de la vidéo correspondant au lien url pour que la fonction `summarize_and_translate(text)` demande au modèle d'effectuer les tâches pour résumer la vidéo
