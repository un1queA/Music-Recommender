Hello! 
This is an AI Music Recommender that helps recommend and suggest artist and songs based on the given genre. The app uses langchain, deepseek to recommend 3 songs from the given genre. It will NOT recommend artists that have already been mentioned in the same session and will also be able to suggest songs with different languages. The app will suggest songs that can be found from the artist's offical youtube channel. However, do note that since I am using deepseek, it only has information until 2024 so artist emerging after that will NOT be suggested as well. I have provided the source code and requirements via requirements.txt.

Since the app requires a deepseek api key to run, I have created an alternate version via ollama which can be installed and run on your own local machine. However, do note that this might result in slower run times as ollama soley relies on your local machine whether it is using its cpu or gpu to run the LLM. I am also unable to use deepseek as my LLM within ollama as it is unable to carry out the task needed for the app to work. Please feel free to try your hand on other Ollama models, the one I have tried specifically is gemma3n:e4b. The code of it can be found via IAMUSIC(OLLAMA).

For running on Ollama, 
# Install Python packages
pip install langchain-community youtube-search streamlit langchain_core

# Install Ollama (follow instructions at https://ollama.com)

# Download DeepSeek model
ollama pull gemma3n:e4b

# Run Ollama in the background
ollama serve (when doing this make sure no Ollama instances are running in the background)

Click on initialize Ollama Connection on the app. Once done, you are ready to go!

In order to run it on your local machine make sure to have the #prerequisites installed on your machine and run it in command prompt with admin privileges.  Do note that that the app is made using the current versions of langchain, openai, youtubesearch function etc. This is not made to be scalable and the code might not work due to version deprecation.
If thats the case make sure to have the latest versions of #prerequisties installed!


<<APP WORKFLOW >>
How the app works are as follows:
User inputs a genre.
App uses Deepseek as an LLM to find an artist in that genre who has an official YouTube channel and hasn't been suggested before.
It then sort of locks the channel to restrict the LLM from only being able to pick videos from that channel only. 
This forces the LLM to scrape that channel's content and pick the three most likely music videos from what's actually there. 
This ensures that the LLM will not pick up videos unrelated to the songs or artist.


Have fun!



# 🎵 AI Music Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://music-recommender-ry79jwgtlxcobwwc7yzydz.streamlit.app)

An intelligent music discovery platform that uses AI to recommend artists and songs based on your favorite genres, complete with YouTube music video integration.

## ✨ Features

- **AI-Powered Recommendations**: Get personalized artist suggestions using Deepseek
- **YouTube Integration**: Direct links to official music videos
- **Genre Exploration**: Discover new music across any genre
- **Smart Memory**: Avoids repeating previously suggested artists
- **Real-time Results**: Instant recommendations as you type

## 🚀 Live Demo




