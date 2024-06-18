from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import parse_qs, urlparse
from openai import OpenAI
from bs4 import BeautifulSoup
import newspaper
import requests
import PIL.Image
import io
import os
import google.generativeai as genai

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# OpenAI configuration
YOUR_API_KEY = "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"
client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

# Google Generative AI configuration
os.environ['GOOGLE_API_KEY'] = "AIzaSyB6wn3odWjs3JjVjjikMhL5fr-8rKyd_WA"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro-vision')

def get_video_id(youtube_url):
    if 'youtube.com/shorts/' in youtube_url:
        return youtube_url.split('/')[-1]
    else:
        query = urlparse(youtube_url).query
        return parse_qs(query)['v'][0]

def extract_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  

        article = newspaper.Article(url)
        article.download()
        article.parse()
        text = article.text

        if text:
            return text

        soup = BeautifulSoup(response.content, 'html.parser')
        article_body = soup.find('article') or soup.find(id='main-content') or soup.find('div', class_='article-body') 

        if article_body:
            all_paragraphs = article_body.find_all('p')
            text = ' '.join([p.text for p in all_paragraphs])
            return text

        return "Unable to extract article content with available methods."

    except requests.exceptions.RequestException as e:
        return f"Error fetching article: {e}"
    except Exception as e:
        return f"Error processing article: {e}"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/fact_check", response_class=HTMLResponse)
def fact_check_page(request: Request):
    return templates.TemplateResponse("fact_check.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit(request: Request, url: str = Form(None), file: UploadFile = File(None)):
    text = ""
    if url:
        if "youtube.com" in url or "youtu.be" in url:
            video_id = get_video_id(url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text_list = [item['text'] for item in transcript]
            text = ' '.join(text_list)
        else:
            text = extract_article_text(url)
    elif file:
        content = await file.read()
        image = PIL.Image.open(io.BytesIO(content))
        response = model.generate_content([
            '''
            Extract the text from the image.
            ''', image])
        text = response.text

    # Send the text to OpenAI Perplexity model for fact-checking
    prompt = f'''I will provide you with news related stories or topics and you will 
                                critically assess the accuracy of the information. 
                                Verify EVERY SINGLE statement, do not skip anything.
                                You should use your own experiences, 
                                thoughtfully explain why something is important, back up claims with facts, 
                                and provide the correct information for any inaccuracies presented in the given information. 
                                The format of your response should be as follows: "Statement 1: ..., 
                                Verdict:(overall accuracy level along with explanation) ... and so on, 
                                (And at the end of the response) Sources links: ...:\n\n{text}'''
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that helps with fact-checking. "
                "Your goal is to provide accurate, concise, and precise responses."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]
    response = client.chat.completions.create(
        model="llama-3-sonar-large-32k-online",
        messages=messages,
    )
    query_result = response.choices[0].message.content

    # Format the result for better readability
    formatted_result = query_result.replace("\n", "\n\n")

    return templates.TemplateResponse("result.html", {"request": request, "query_result": formatted_result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
