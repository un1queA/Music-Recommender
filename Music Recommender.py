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
    """Generate artist and song recommendations with duplicate prevention and video availability check"""
    
    # Import here to avoid circular imports
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    
    llm = initialize_llm()
    if not llm:
        st.error("LLM not initialized. Please check your API key.")
        return None

    # Define prompt templates - UPDATED to request songs with official music videos
    prompt_template_genre = PromptTemplate(
        input_variables=["genre"],
        template="""I am trying to discover new music. Give me artists or singers that specialize in {genre}. 
        Suggest only one artist. 
        IMPORTANT: 
        - Use the artist's native/original name, not English translations
        - For international artists, use their name in the original language (e.g., 宇多田ヒカル not Utada Hikaru, BTS not Bangtan Boys)
        - Avoid duplicates or similar names to previous suggestions
        - Please be creative and provide any artist ranging from famous to niche!"""
    )
    
    prompt_template_song = PromptTemplate(
        input_variables=["artist_suggestion"],
        template="""Give me 3 popular songs from {artist_suggestion} that have official music videos on YouTube. 
        IMPORTANT: 
        1. Only suggest songs that have official music videos
        2. Use the original song titles in their native language - DO NOT translate them to English
        3. Avoid songs that only have lyric videos, live performances, or fan-made content
        4. Return only the song titles, one per line
        5. Make sure the songs are well-known and likely to have official videos
        
        Examples:
        - If the artist is BTS, use Korean titles like "Dynamite" not translations
        - If the artist is Bad Bunny, use Spanish titles like "Tití Me Preguntó" not English translations
        - If the artist is Lisa (Thai singer), use original titles like "LALISA" not translations
        - If the artist is 宇多田ヒカル, use Japanese titles like "First Love" not translations"""
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

def find_artist_official_channel(artist: str) -> str:
    """Find the official YouTube channel for an artist using native name"""
    try:
        # Search for the artist's official channel using native name
        channel_queries = [
            f"{artist} official",
            f"{artist} topic",
            f"{artist} vevo",
            f"{artist}",
            f'"{artist}" official'  # Exact phrase for non-English names
        ]
        
        for query in channel_queries:
            results = YoutubeSearch(query, max_results=5).to_dict()  # Increased results
            for result in results:
                channel_name = result['channel'].lower()
                artist_lower = artist.lower()
                
                # Strong indicators of official channel
                if (artist_lower in channel_name or 
                    'official' in channel_name or 
                    'vevo' in channel_name or
                    'topic' in channel_name):
                    return result['channel']  # Return the channel name
        
        return None
    except Exception as e:
        return None  # Silent fail, will use fallback

def search_in_channel(artist: str, song: str, channel_name: str) -> str:
    """Search for a song within a specific channel"""
    try:
        # Use original song title without translation or cleaning for non-English songs
        search_queries = [
            f"{song} {artist}",
            f"{song} official",
            f"{song} music video",
            f"{song}",
            f'"{song}"',  # Exact phrase search for non-English titles
            f"{song} mv"  # Common abbreviation for music video
        ]
        
        for query in search_queries:
            # Search with channel context
            full_query = f"{query} {channel_name}"
            results = YoutubeSearch(full_query, max_results=5).to_dict()  # More results
            
            for result in results:
                # Check if the result is from the right channel
                if result['channel'].lower() == channel_name.lower():
                    video_title = result['title'].lower()
                    # Higher confidence since it's from official channel
                    if not any(term in video_title for term in ['lyric', 'cover', 'karaoke', 'tribute', 'fanmade']):
                        return f"https://www.youtube.com/watch?v={result['id']}"
        
        return None
    except Exception as e:
        return None

def youtube_search(artist: str, song: str) -> str:
    """Search for official music video on YouTube with channel-first approach"""
    try:
        # Use original song title - don't lowercase or clean non-English characters
        artist_lower = artist.lower().strip()
        # Keep song in original form for non-English titles
        
        # STEP 1: Try to find official channel first
        official_channel = find_artist_official_channel(artist)
        official_video_found = False
        
        # STEP 2: If official channel found, search within that channel
        if official_channel:
            st.info(f"🎵 Found official channel: {official_channel}")
            channel_video = search_in_channel(artist, song, official_channel)
            if channel_video:
                official_video_found = True
                return channel_video
        
        # STEP 3: Fallback to comprehensive search with multiple strategies
        search_queries = [
            f'"{artist}" "{song}" official music video',  # Exact phrase for non-English
            f"{artist} {song} official music video",
            f"{artist} {song} music video", 
            f"{song} {artist} music video",
            f'"{song}" {artist}',  # Exact song title
            f"{artist} {song} official",
            f"{song} {artist} official",
            f"{artist} {song}",
            f"{song} {artist}",
            f"{song} mv {artist}",  # Music video abbreviation
        ]
        
        best_official_result = None
        best_fallback_result = None
        best_official_score = 0
        best_fallback_score = 0
        
        for query in search_queries:
            try:
                results = YoutubeSearch(query, max_results=8).to_dict()  # More results per query
                
                for result in results:
                    video_title = result['title'].lower()
                    channel_name = result['channel'].lower()
                    video_id = result['id']
                    
                    # Calculate a relevance score for this result
                    score = calculate_relevance_score(video_title, channel_name, artist_lower, song)
                    
                    # Check if this is from official channel (highest priority)
                    is_official_channel = (
                        artist_lower in channel_name or 
                        'official' in channel_name or 
                        'vevo' in channel_name
                    )
                    
                    # Check if it's the same artist singing (avoid covers)
                    is_same_artist = (
                        artist_lower in video_title or 
                        any(word in video_title for word in artist_lower.split())
                    )
                    
                    # STRONG penalty for covers and unwanted content
                    unwanted_penalty = 0
                    unwanted_terms = ['cover', 'tribute', 'fan', 'karaoke', 'lyric', 'lyrics']
                    for term in unwanted_terms:
                        if term in video_title:
                            unwanted_penalty = 0.5  # Strong penalty
                            break
                    
                    final_score = score - unwanted_penalty
                    
                    # Prioritize official channel videos
                    if is_official_channel and final_score > best_official_score:
                        best_official_score = final_score
                        best_official_result = video_id
                    
                    # Fallback: same artist singing the same song (even if not official)
                    elif is_same_artist and final_score > best_fallback_score and unwanted_penalty == 0:
                        best_fallback_score = final_score
                        best_fallback_result = video_id
                    
                    # If we find a very good official match, return immediately
                    if is_official_channel and final_score >= 0.8:
                        return f"https://www.youtube.com/watch?v={video_id}"
                        
            except Exception as query_error:
                # Continue with next query if one fails
                continue
        
        # Return priority: official channel videos first, then same artist fallbacks
        if best_official_result and best_official_score >= 0.5:
            return f"https://www.youtube.com/watch?v={best_official_result}"
        elif best_fallback_result and best_fallback_score >= 0.4:
            st.info(f"🎵 Found performance by {artist} (may not be official)")
            return f"https://www.youtube.com/watch?v={best_fallback_result}"
        else:
            return None
            
    except Exception as e:
        st.error(f"YouTube search error: {e}")
        return None
    
def calculate_relevance_score(video_title: str, channel_name: str, artist: str, song: str) -> float:
    """Calculate how relevant a YouTube video is to the requested artist and song"""
    score = 0.0
    
    # Clean artist but preserve song title for non-English characters
    clean_artist = re.sub(r'[^\w\s]', '', artist).strip()
    # Don't clean song title to preserve non-English characters
    clean_video_title = re.sub(r'[^\w\s]', '', video_title).strip()
    
    # 1. Check if artist is in channel name (very important)
    if clean_artist in channel_name:
        score += 0.5  # Higher weight for official channels
    elif 'official' in channel_name and any(word in channel_name for word in clean_artist.split()):
        score += 0.4
    elif 'vevo' in channel_name:
        score += 0.3

    # 2. Check if song title appears in video title - use original song title
    song_lower = song.lower()
    if song_lower in video_title:
        score += 0.4  # Higher weight for exact match with original title
    else:
        # For non-English titles, use partial matching
        song_words = set(song_lower.split())
        title_words = set(video_title.split())
        
        matching_words = song_words.intersection(title_words)
        if len(song_words) > 0:
            title_match_ratio = len(matching_words) / len(song_words)
            score += title_match_ratio * 0.3
    
    # 3. Check for official indicators in title
    if 'official' in video_title and 'music video' in video_title:
        score += 0.2
    elif 'official' in video_title:
        score += 0.15
    elif 'music video' in video_title or 'mv' in video_title:
        score += 0.1
    
    # 4. Check if artist appears in video title
    if any(word in clean_video_title for word in clean_artist.split()):
        score += 0.15
    
    # 5. Bonus for exact matches
    if f"{clean_artist} {song_lower}" in video_title:
        score += 0.25
    
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
            videos_found = 0
            for song in songs:
                st.write(f"- {song}")
                
                # Get YouTube link for each song with improved search
                with st.spinner(f"Searching for '{song}'..."):
                    video_url = youtube_search(artist, song)
                
                if video_url:
                    st.video(video_url)
                    st.success(f"✅ Found video for '{song}'")
                    videos_found += 1
                else:
                    st.warning(f"❌ No suitable video found for '{song}'")
            
            if videos_found > 0:
                st.success(f"Done! Found {videos_found} out of {len(songs)} videos.")
            else:
                st.error("No videos found. Try a different genre or artist.")

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
