#prerequisites:
#pip install langchain
#pip install openai
#pip install langchain -U langchain-community
#pip install youtube-search
#pip install httpx==0.24.1
#pip install streamlit
#pip install streamlit -U streamlit
#pip install -U langchain-openai
#Run with this command ==> python -m streamlit run IAMUSIC2.py --logger.level=error to run the script with error logging level

import streamlit as st
import os
import logging
import re #regular expressions for cleaning artist names
from youtube_search import YoutubeSearch
import time
# Set up logging
logging.getLogger("streamlit").setLevel(logging.WARNING)

# =============================================================================
# BACKEND FUNCTIONS
# =============================================================================

def validate_api_key(api_key):
    """Check if the API key is a valid OpenAI GPT key"""
    if not api_key.startswith('sk-'):
        return False
    # For now, we'll just check the pattern. You can add actual API validation later.
    return True

def clean_artist_name(artist):
    """Clean artist name to avoid duplicates like 'lisa' and 'lisa (japanese singer)'"""
    # Remove anything in parentheses and extra spaces
    clean_name = re.sub(r'\([^)]*\)', '', artist).strip() #removes any text within parentheses along with the parentheses themselves, then trims leading/trailing spaces
    #This prevents "Lisa" and "Lisa (Japanese singer)" from being treated as different artists.
    # Convert to lowercase for comparison
    return clean_name.lower()

def initialize_llm():
    """Initialize the LLM with the current API key"""
    try:
        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4", 
            temperature=1.0,
            openai_api_key=os.environ.get("OPENAI_API_KEY")  # Pass the API key directly
        ) #using chatgpt4 as our LLM (Large Language Model) for generating responses, the temperature indicates how creative/random the response from chatgpt will be
        return llm
    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {e}")
        return None

def generate_artist_and_songs(genre):
    """Generate artist and song recommendations with duplicate prevention"""
    
    # Import here to avoid circular imports
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    
    llm = initialize_llm()
    if not llm:
        st.error("LLM not initialized. Please check your API key.")
        return None

    # Define prompt templates
    prompt_template_genre = PromptTemplate(
        input_variables=["genre"],
        template="I am trying to discover new music. Give me artists or singers that specialize in {genre}. Suggest only one artist. Avoid duplicates or similar names to previous suggestions. Please be creative and provide any artist ranging from famous to niche!"
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
        clean_artist = clean_artist_name(artist)
        
        if clean_artist not in st.session_state.used_artists:
            st.session_state.used_artists.add(clean_artist)
            return response  # Return the successful response immediately
#allows chatgpt to try 5 times to find a new artist, if it finds a new artist it will return the response immediately, if not it will try again. At the same time it will store the artist that has been found and also check if the artist genertated is a duplicate or not, if it is a duplicate it will try again until it finds a new artist or reaches the maximum number of attempts.        

    # If we get here, all attempts failed
    st.error("Couldn't find a new artist after 5 attempts. Try a different genre or reset memory.")
    return None


def youtube_search(artist: str, song: str) -> str:
    """Search for official music video on YouTube"""
    try:
        # Clean the song name by removing common symbols and extra words
        clean_song = re.sub(r'[^\w\s]', '', song).lower()
        # Try multiple search queries to find official video
        search_queries = [
            f"{artist} {song} official music video",
            f"{song} {artist} official music video",
            f"{artist} {song} official video",
             f"{artist} {song} official music video",
            f"{song} {artist}",
            f"{artist} {song}",
        ]
        

        best_result = None
        best_score = 0
        best_title = ""
        
        for query in search_queries:
            results = YoutubeSearch(query, max_results=5).to_dict()
            
            for result in results:
                video_title = result['title'].lower()
                channel_name = result['channel'].lower()
                video_id = result['id']
                
                # Calculate a relevance score for this result
                score = calculate_relevance_score(video_title, channel_name, artist, clean_song)
                
                # If we find a perfect match, return immediately
                if score >= 0.9:
                    return f"https://www.youtube.com/watch?v={video_id}"
                
                # Track the best result so far
                if score > best_score:
                    best_score = score
                    best_result = video_id
                    best_title = video_title
        
        # Return the best result if it meets minimum threshold
        if best_result and best_score >= 0.6:  # At least 60% match
            return f"https://www.youtube.com/watch?v={best_result}"
        else:
            return None
            
    except Exception as e:
        st.error(f"YouTube search error: {e}")
        return None
    
def calculate_relevance_score(video_title: str, channel_name: str, artist: str, clean_song: str) -> float:
    """Calculate how relevant a YouTube video is to the requested artist and song"""
    score = 0.0
    
    
    # 1. Check if artist is in channel name (very important)
    if artist in channel_name:
        score += 0.5
    elif 'official' in channel_name or artist in channel_name:
        score += 0.3

    # 2. Check if song title appears in video title
    clean_video_title = re.sub(r'[^\w\s]', '', video_title)
    
    # Check for exact song title match
    if clean_song in clean_video_title:
        score += 0.3
    else:
        # Word-by-word matching as fallback
        song_words = set(clean_song.split())
        title_words = set(clean_video_title.split())
        
        matching_words = song_words.intersection(title_words)
        if len(song_words) > 0:
            title_match_ratio = len(matching_words) / len(song_words)
            score += title_match_ratio * 0.2
    
    # 3. Check for official indicators in title
    if 'official' in video_title and 'music video' in video_title:
        score += 0.2
    
    # 4. Penalize for unwanted content types
    unwanted_terms = ['lyric', 'lyrics', 'cover', 'tribute', 'fan', 'karaoke', 'instrumental']
    for term in unwanted_terms:
        if term in video_title:
            score -= 0.4
            break
    
   
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))

#Takes the artist and song name to search for its offical music video on Youtube, if it finds the video it will return the link to the video, if not it will return None. The limit is set to 1 to only get the first result.

# =============================================================================
# FRONTEND FUNCTIONS
# =============================================================================

def setup_api_key():
    """Handle API key setup"""
    # Check if we should hide the API key section (after successful validation)
    if st.session_state.get('api_key_valid') and st.session_state.get('hide_api_section', False):
        return
    
    st.sidebar.header("🔑 API Configuration")
    
    # Check if API key exists in environment
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and validate_api_key(env_key):
        os.environ["OPENAI_API_KEY"] = env_key
        st.session_state.api_key_valid = True
        st.sidebar.success("Using API key from environment")
        # Set a timer to hide the section
        if 'hide_api_section' not in st.session_state:
            st.session_state.hide_api_section = True
        return
    
    # User input for API key
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    if st.sidebar.button("Validate Key"):
        if validate_api_key(api_key):
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.api_key_valid = True
            st.session_state.hide_api_section = True  # Mark to hide section
            st.sidebar.success("✅ API Key validated!")
            
            # Add a small delay and then rerun to hide the section
            time.sleep(2)  # Show success message for 2 seconds
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid API key")

def main_app():
    """Main application interface"""
    # Streamlit UI (HOME PAGE)
    st.title("🎵 I AM MUSIC")
    genre = st.text_input("Enter a genre of music:")

    if genre:
        with st.spinner("Generating recommendations..."):
            # Get response from LangChain
            response = generate_artist_and_songs(genre)
            
            if response is None:
                return
                
            artist = response["artist_suggestion"]
            songs = [s.strip().replace('- ', '').replace('• ', '') 
                    for s in response["song_suggestion"].split('\n') 
                    if s.strip() and len(s.strip()) > 2]

    #The code processes it as:
    #split('\n') creates: ["- Song 1", "• Song 2", "Song 3", "", "- Short"]
    # if s.strip() and len(s.strip()) > 2 filters out empty lines and very short text
    # For each remaining line:
    # s.strip() removes spaces: "- Song 1" → "- Song 1"
    # .replace('- ', '') removes bullet: "- Song 1" → "Song 1"
    # .replace('• ', '') removes bullet: "• Song 2" → "Song 2"
    # Final result: ["Song 1", "Song 2", "Song 3"]
    # The line "- Short" gets filtered out because after cleaning it becomes "Short" which has only 5 characters, but if your condition is len(s.strip()) > 2, it would be included. If you meant to filter out very short song names, you might want > 3 or > 4.       
            
            # Display results
            st.subheader(f"Recommended Artist: {artist}")
            
            st.subheader("Recommended Songs:")
            for song in songs:
                st.write(f"- {song}")
                
                # Get YouTube link for each song
                if video_url := youtube_search(artist, song):  # Fixed: pass both artist and song
                    st.video(video_url)
                else:
                    st.warning("Video not found")
            
            st.success("Done!")

# =============================================================================
# APP INITIALIZATION AND MAIN LOGIC
# =============================================================================

# Initialize manual tracking
if 'used_artists' not in st.session_state: #store the used artists in the session state to avoid repeating them, in this case we are using streamlit's session state
    st.session_state.used_artists = set()

# Initialize API key management
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

# Setup API key first
setup_api_key()

# Only run main app if API key is valid
if not st.session_state.api_key_valid:
    st.info("🔑 Please enter your OpenAI API key in the sidebar to use the app.")
else:
    main_app()


