import re

def extract_video_id(url):
    """Extracts video ID from a YouTube URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

import requests

YOUTUBE_API_KEY = "your api key "

def get_youtube_comments(video_id, max_results=10):
    """Fetches comments from a YouTube video."""
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={YOUTUBE_API_KEY}&maxResults={max_results}"
    response = requests.get(url).json()

    comments = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
        comments.append(comment)
    
    return comments
