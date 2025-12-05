import streamlit as st
import os
import re
from typing import List, Dict, Optional
from youtube_search import YoutubeSearch

# Import LangChain components
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =============================================================================
# LLM CHAINS & PROMPT TEMPLATES
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with the provided API key"""
    try:
        llm = ChatOpenAI(
            model="gpt-4", 
            temperature=0.7,
            openai_api_key=api_key,
            max_tokens=1000
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {e}")
        return None

def create_artist_finder_chain(llm: ChatOpenAI):
    """Create chain to find artist with official YouTube channel"""
    
    artist_prompt = PromptTemplate(
        input_variables=["genre"],
        template="""
        You are a music expert. Recommend ONE artist who is known for {genre} music.
        
        IMPORTANT:
        1. The artist should have a YouTube channel (most artists do)
        2. The artist should be well-known enough to have official music videos
        3. Genre can be in any language (K-pop, J-pop, Reggaeton, etc.)
        
        Return EXACTLY in this format:
        ARTIST: [Full Artist Name]
        DESCRIPTION: [Brief description]
        """
    )
    
    return LLMChain(llm=llm, prompt=artist_prompt, output_key="artist_info")

def create_song_finder_chain(llm: ChatOpenAI):
    """Create chain to find songs"""
    
    song_prompt = PromptTemplate(
        input_variables=["artist_info", "genre"],
        template="""
        Based on this artist: {artist_info}
        
        Suggest 3 popular songs by this artist in the {genre} genre.
        These should be songs that likely have official music videos on YouTube.
        
        Return EXACTLY in this format:
        SONG 1: [Song Title 1]
        SONG 2: [Song Title 2]
        SONG 3: [Song Title 3]
        """
    )
    
    return LLMChain(llm=llm, prompt=song_prompt, output_key="songs_info")

# =============================================================================
# YOUTUBE SEARCH FUNCTIONS
# =============================================================================

def search_youtube_for_song(artist: str, song: str) -> Optional[str]:
    """Search YouTube for a song by an artist"""
    try:
        # Try multiple search patterns
        search_patterns = [
            f"{artist} {song} official music video",
            f"{artist} {song} official",
            f"{song} {artist} official",
            f"{artist} {song}",
        ]
        
        for query in search_patterns:
            results = YoutubeSearch(query, max_results=3).to_dict()
            
            if results:
                # Return first result (usually most relevant)
                video_id = results[0]['id']
                return f"https://www.youtube.com/watch?v={video_id}"
                
    except Exception as e:
        st.warning(f"YouTube search error: {str(e)[:100]}")
    
    return None

def extract_songs_from_llm_output(songs_output: str) -> List[str]:
    """Extract song titles from LLM output"""
    songs = []
    lines = songs_output.strip().split('\n')
    
    for line in lines:
        # Remove numbering and bullets
        clean_line = re.sub(r'^SONG\s*\d+:\s*', '', line.strip())
        clean_line = re.sub(r'^[•\-]\s*', '', clean_line)
        
        if clean_line and len(clean_line) > 2:
            songs.append(clean_line)
    
    return songs[:3]  # Return max 3 songs

def extract_artist_from_output(artist_output: str) -> tuple:
    """Extract artist name and description from LLM output"""
    artist_name = ""
    artist_desc = ""
    
    lines = artist_output.strip().split('\n')
    for line in lines:
        if line.startswith("ARTIST:"):
            artist_name = line.replace("ARTIST:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            artist_desc = line.replace("DESCRIPTION:", "").strip()
    
    return artist_name, artist_desc

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="🎵 Music Discovery",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # Title and description
    st.title("🎵 Music Discovery")
    st.markdown("""
    Discover music in any genre! The AI suggests artists and songs, then we search YouTube for the videos.
    """)
    
    # API Key input
    st.sidebar.header("🔑 API Configuration")
    api_key = st.sidebar.text_input(
        "OpenAI API Key:",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/api-keys",
        key="api_key_input"
    )
    
    # Store API key
    if api_key:
        st.session_state.api_key = api_key
    
    # Genre input
    genre = st.text_input(
        "Enter a music genre:",
        placeholder="e.g., K-pop, Rock, Hip Hop, Jazz, Classical, Reggae...",
        key="genre_input"
    )
    
    # Search button
    if st.button("🎯 Find Music", type="primary", use_container_width=True):
        if not genre:
            st.error("Please enter a music genre!")
            return
        
        if not st.session_state.api_key or not st.session_state.api_key.startswith('sk-'):
            st.error("Please enter a valid OpenAI API key in the sidebar!")
            return
        
        with st.spinner(f"Searching for {genre} music..."):
            try:
                # Initialize LLM
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    st.error("Failed to initialize OpenAI. Check your API key.")
                    return
                
                # Create chains
                artist_chain = create_artist_finder_chain(llm)
                song_chain = create_song_finder_chain(llm)
                
                # Create sequential chain
                full_chain = SequentialChain(
                    chains=[artist_chain, song_chain],
                    input_variables=["genre"],
                    output_variables=["artist_info", "songs_info"],
                    verbose=False
                )
                
                # Run the chain
                result = full_chain({"genre": genre})
                
                # Extract information
                artist_name, artist_desc = extract_artist_from_output(result["artist_info"])
                songs = extract_songs_from_llm_output(result["songs_info"])
                
                if not artist_name or not songs:
                    st.error("Could not parse the AI's response. Please try again.")
                    return
                
                # Display results
                st.success(f"✅ Found: **{artist_name}**")
                
                if artist_desc:
                    st.info(f"*{artist_desc}*")
                
                st.markdown("---")
                st.subheader("🎵 Recommended Songs")
                
                # Search for and display each song
                for i, song_title in enumerate(songs, 1):
                    with st.spinner(f"Searching for '{song_title}'..."):
                        youtube_url = search_youtube_for_song(artist_name, song_title)
                    
                    # Create columns for layout
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {song_title}**")
                    
                    with col2:
                        if youtube_url:
                            st.markdown(
                                f'<a href="{youtube_url}" target="_blank">'
                                '<button style="background-color: #FF0000; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;">▶ Watch</button></a>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("Video not found")
                    
                    # Display video if found
                    if youtube_url:
                        try:
                            st.video(youtube_url)
                        except:
                            st.markdown(f"[Open video in YouTube]({youtube_url})")
                    
                    st.markdown("---")
                
                # Store in history
                st.session_state.recommendations.append({
                    "genre": genre,
                    "artist": artist_name,
                    "songs": songs,
                    "timestamp": "Now"
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display previous searches
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Search History")
        
        for i, rec in enumerate(reversed(st.session_state.recommendations[-5:]), 1):
            with st.sidebar.expander(f"{rec['artist']} ({rec['genre']})"):
                st.write(f"**Genre:** {rec['genre']}")
                st.write(f"**Songs:**")
                for song in rec['songs']:
                    st.write(f"- {song}")
    
    # How it works
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **This app works in 3 steps:**
        
        1. **AI Suggestion**: ChatGPT suggests an artist and 3 songs based on the genre
        2. **YouTube Search**: We search YouTube for those specific songs
        3. **Display Results**: Show the videos with clickable links
        
        **Important Notes:**
        - ChatGPT cannot browse the internet itself
        - It uses knowledge from its training (up to 2023)
        - We use separate YouTube search to find current videos
        - Some niche songs might not have YouTube videos
        
        **Try these genres:**
        - K-pop (BTS, Blackpink)
        - Reggaeton (Bad Bunny, J Balvin)
        - City Pop (Mariya Takeuchi)
        - Lo-fi Hip Hop
        - Classical Piano
        """)

if __name__ == "__main__":
    main()
