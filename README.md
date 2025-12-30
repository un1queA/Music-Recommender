Hello! 
This is an AI Music Recommender that helps recommend and suggest artist and songs based on the given genre. The app uses langchain, deepseek to recommend 3 songs from the given genre. It will NOT recommend artists that have already been mentioned in the same session and will also be able to suggest songs with different languages. The app will suggest songs that can be found from the artist's offical youtube channel. However, do note that since I am using deepseek, it only has information until 2024 so artist emerging after that will NOT be suggested as well. I have provided the source code and requirements via requirements.txt.

Since the app requires a deepseek api key to run which requires a fee of at least 5 USD to use it, I have created an alternate version via ollama which can be installed and run on your own local machine. However, do note that this might result in slower run times as ollama soley relies on your local machine whether it is using its cpu or gpu to run the LLM. I am also unable to use deepseek as my LLM within ollama as it is unable to carry out the task needed for the app to work. Please feel free to try your hand on other Ollama models, the one I have tried specifically is gemma3n:e4b. The code of it can be found via IAMUSIC(OLLAMA).

# APP IN DEPTH BREAKDOWN
IAMUSIC is an AI Music Recommender that helps recommend and suggest artist and songs based on the given genre. It has to be able to recommend 3 songs from the given genre. It will NOT recommend artists that have already been mentioned in the same session and will also be able to suggest songs from different languages(this should only be the case if the genre is of a different language other than english). The app will suggest songs that can be found from the artist's offical youtube channel. However, do note that since I am using deepseek api, it only has information until 2024 so artist emerging after that will NOT be suggested as well. The code is universal so there is no hardcoding at all. The code should accept any genre no matter the language or niche and provide an artist with an official youtube channel. After finding the channel, lock the channel so that the songs cannot be chosen out of the locked chanel and only from the channel that has been locked onto not anything else. The app should be able to recommend up to at least 50 artists if the genre is popular and should only be unable to do so if the genre is too niche. If IAMUSIC is unable to recommend 3 unique and different songs from the same artist that has not been mentioned during the session despite a popular genre it should always retry.

# Running the app free via Ollama

For running on Ollama,

Make sure you have the following Python packages installed if not type the following into command prompt. Take note all of this should be done on your local machine
pip install langchain-community youtube-search streamlit langchain_core

Install Ollama (follow instructions at https://ollama.com)

Download LLM model, choose any model that is present on https://ollama.com and replace the model with gemma3n:e4b. The command to download the LLM is as follows:
ollama pull gemma3n:e4b

Run Ollama in the background
ollama serve (when doing this make sure no Ollama instances are running in the background)

Click on initialize Ollama Connection on the app. Once done, you are ready to go!






In order to run it on your local machine make sure to have the #prerequisites installed on your machine and run it in command prompt with admin privileges alterantively you can click on this button to try it out on streamlit community cloud :) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://music-recommender-ry79jwgtlxcobwwc7yzydz.streamlit.app).  Do note that that the app is made using the current versions of langchain, openai, youtubesearch function etc. This is not made to be scalable and the code might not work due to version deprecation.
If thats the case make sure to have the latest versions of #prerequisties installed!
Have fun!







