#prerequisites:
#pip install langchain
#pip install openai
#pip install langchain -U langchain-community
#pip install youtube-search
#pip install httpx==0.24.1
#pip install streamlit
#pip install streamlit -U streamlit
#pip install -U langchain-openai
#Run with this command ==> python -m streamlit run LLM1.py --logger.level=error to run the script with error logging level
import streamlit as st
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI #import the OpenAI package that we just installed
from langchain.prompts import PromptTemplate #import the PromptTemplate which is used to create a template for the prompt that we will send to the AI
from langchain.chains import LLMChain #import the LLMChain which is used to create a chain of prompts and responses
from langchain.chains import SequentialChain #import the SequentialChain which is used to create a chain of prompts and responses in a sequential manner
from langchain.memory import ConversationBufferMemory #import the ConversationBufferMemory which is used to store the conversation history
from langchain.chains import ConversationChain #import the ConversationChain which is used to create a chain of prompts and responses in a conversational manner
from youtube_search import YoutubeSearch
from langchain.agents import AgentType, initialize_agent, load_tools #TAKE NOTE havent implemented this yet

# Set up logging
logging.getLogger("streamlit").setLevel(logging.WARNING)

# Set up OpenAI API key 
openapi_key = ""  # Replace with a valid key
os.environ["OPENAI_API_KEY"] = openapi_key

# Initialize manual tracking
if 'used_artists' not in st.session_state: #store the used artists in the session state to avoid repeating them, in this case we are using streamlit's session state
    st.session_state.used_artists = set()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=1.0) #using chatgpt4 as our LLM (Large Language Model) for generating responses, the temperature indicates how creative/random the response from chatgpt will be


# Streamlit UI (HOME PAGE)
st.title("🎵 I AM MUSIC")
genre = st.text_input("Enter a genre of music:")

def generate_artist_and_songs(genre):
    # Define prompt templates
    prompt_template_genre = PromptTemplate(
        input_variables=["genre"],
        template="I am trying to discover new music. Give me artists or singers that specialize in {genre}. Suggest only one."
    )
    
    prompt_template_song = PromptTemplate(
        input_variables=["artist_suggestion"],
        template="Give me 3 songs from {artist_suggestion}."
    )

    # Create chains
    genre_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_genre,
        output_key="artist_suggestion"
    )

    song_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_song,
        output_key="song_suggestion"
    )

    # Create sequential chain
    chain = SequentialChain(
        chains=[genre_chain, song_chain],
        input_variables=["genre"],
        output_variables=["artist_suggestion", "song_suggestion"],
        verbose=False  # Set to False to reduce console output
    )
#combine the two chains into a sequential chain, the first chain will generate an artist based on the genre and the second chain will generate songs based on the artist, the prompts can be seen via template=

    max_attempts = 5
    for attempt in range(max_attempts):
        response = chain({"genre": genre})
        artist = response["artist_suggestion"].strip()
        
        if artist not in st.session_state.used_artists:
            st.session_state.used_artists.add(artist)
            return response  # Return the successful response immediately
#allows chatgpt to try 5 times to find a new artist, if it finds a new artist it will return the response immediately, if not it will try again. At the same time it will store the artist that has been found and also check if the artist genertated is a duplicate or not, if it is a duplicate it will try again until it finds a new artist or reaches the maximum number of attempts.        

    # If we get here, all attempts failed
    st.error("Couldn't find a new artist after 5 attempts. Try a different genre or reset memory.")
    st.stop()

# Function to search for YouTube videos
def youtube_search(query: str) -> str:
    try:
        results = YoutubeSearch(f"{query} official music video", max_results=1).to_dict()
        if results:
            return f"https://www.youtube.com/watch?v={results[0]['id']}"
        return None
    except Exception as e:
        st.error(f"YouTube search error: {e}")
        return None

#Takes the artist and song name to search for its offical music video on Youtube, if it finds the video it will return the link to the video, if not it will return None. The limit is set to 1 to only get the first result.

#Streamlit UI
if genre:
    with st.spinner("Generating recommendations..."):
        # Get response from LangChain
        response = generate_artist_and_songs(genre)
        artist = response["artist_suggestion"]
        songs = [s.strip() for s in response["song_suggestion"].split('\n') if s.strip()]
        
        
        # Display results
        st.subheader(f"Recommended Artist: {artist}")
        
        st.subheader("Recommended Songs:")
        for song in songs:
            st.write(f"- {song}")
            
            # Get YouTube link for each song
            if video_url := youtube_search(f"{artist} {song}"):
                st.video(video_url)
            else:
                st.warning("Video not found")
        
        st.success("Done!")