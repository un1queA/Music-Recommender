# prerequisites:
# pip install langchain openai langchain-community youtube-search streamlit
# Run: streamlit run app.py

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
        Find an artist who has an OFFICIAL YouTube channel and specializes in {genre} music.
        
        IMPORTANT REQUIREMENTS:
        1. The artist MUST have an official, verified YouTube channel
        2. The artist should be known for {genre} music
        3. The genre can be in any language (not just English)
        4. Provide the exact artist name as it appears on YouTube
        
        Return in this format:
        ARTIST: [Artist Name]
        DESCRIPTION: [Brief description of the artist and their YouTube channel]
        """
    )
    
    return LLMChain(llm=llm, prompt=artist_prompt, output_key="artist_info")

def create_song_finder_chain(llm: ChatOpenAI):
    """Create chain to find songs from official channel"""
    
    song_prompt = PromptTemplate(
        input_variables=["artist_info", "genre"],
        template="""
        Based on this artist information: {artist_info}
        
        Find 3 songs from their OFFICIAL YouTube channel that represent {genre} music.
        
        For EACH song, provide:
        1. The exact song title as it appears on YouTube
        2. A search query that will find the official video on YouTube
        
        Format exactly like this:
        SONG 1 TITLE: [Title]
        SONG 1 SEARCH: [Artist Name] [Song Title] official music video
        
        SONG 2 TITLE: [Title]
        SONG 2 SEARCH: [Artist Name] [Song Title] official video
        
        SONG 3 TITLE: [Title]
        SONG 3 SEARCH: [Artist Name] [Song Title] official
        """
    )
    
    return LLMChain(llm=llm, prompt=song_prompt, output_key="songs_info")

# =============================================================================
# YOUTUBE SEARCH FUNCTIONS
# =============================================================================

def search_youtube_video(search_query: str, artist_name: str) -> Optional[str]:
    """Search for official YouTube video with enhanced filtering"""
    try:
        results = YoutubeSearch(search_query, max_results=10).to_dict()
        
        # Clean artist name for comparison
        clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            
            # Strong indicators of official content
            official_indicators = [
                'official',
                'music video',
                'mv]',
                '【mv】',
                'lyric video',
                'visualizer'
            ]
            
            # Check if channel contains artist name
            artist_in_channel = any(word in channel for word in clean_artist.split())
            
            # Check if title contains official indicators
            is_official = any(indicator in title for indicator in official_indicators)
            
            # Check for common non-official indicators (avoid these)
            non_official_indicators = [
                'cover',
                'tribute',
                'karaoke',
                'instrumental',
                'remix',
                'mashup',
                'fan made',
                'lyrics only'
            ]
            
            is_unofficial = any(indicator in title for indicator in non_official_indicators)
            
            # If it's from the artist's channel and looks official, return it
            if artist_in_channel and is_official and not is_unofficial:
                return f"https://www.youtube.com/watch?v={video_id}"
            
            # Fallback: if artist in channel and not unofficial, accept it
            if artist_in_channel and not is_unofficial:
                return f"https://www.youtube.com/watch?v={video_id}"
        
        # If no perfect match, return first result (with warning)
        if results:
            return f"https://www.youtube.com/watch?v={results[0]['id']}"
            
    except Exception as e:
        st.warning(f"YouTube search error for '{search_query}': {str(e)[:100]}")
    
    return None

def parse_llm_output(artist_output: str, songs_output: str) -> Dict:
    """Parse LLM outputs into structured data"""
    try:
        # Parse artist info
        artist_lines = artist_output.split('\n')
        artist_name = ""
        artist_desc = ""
        
        for line in artist_lines:
            if line.startswith("ARTIST:"):
                artist_name = line.replace("ARTIST:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                artist_desc = line.replace("DESCRIPTION:", "").strip()
        
        # Parse songs info
        songs = []
        lines = songs_output.split('\n')
        
        for i in range(0, len(lines), 3):
            if i + 1 < len(lines):
                title_line = lines[i]
                search_line = lines[i + 1] if i + 1 < len(lines) else ""
                
                if "TITLE:" in title_line:
                    title = title_line.split("TITLE:", 1)[1].strip()
                    
                    # Extract search query
                    search_query = ""
                    if "SEARCH:" in search_line:
                        search_query = search_line.split("SEARCH:", 1)[1].strip()
                    else:
                        search_query = f"{artist_name} {title} official music video"
                    
                    # Search for YouTube video
                    youtube_url = search_youtube_video(search_query, artist_name)
                    
                    if youtube_url:
                        songs.append({
                            "title": title,
                            "youtube_url": youtube_url,
                            "search_query": search_query
                        })
        
        return {
            "artist_name": artist_name,
            "artist_description": artist_desc,
            "songs": songs[:3]  # Limit to 3 songs
        }
        
    except Exception as e:
        st.error(f"Error parsing LLM output: {e}")
        return None

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="🎵 Music Discovery Pro",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # Title and description
    st.title("🎵 Music Discovery Pro")
    st.markdown("""
    Discover artists with official YouTube channels in any music genre.
    Enter a genre below to get started!
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
    
    # Store API key in session state
    if api_key:
        st.session_state.api_key = api_key
    
    # Genre input
    genre = st.text_input(
        "Enter a music genre:",
        placeholder="e.g., K-pop, Lo-fi Hip Hop, Flamenco, Vaporwave, Bhangra...",
        key="genre_input"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button(
            "🎯 Find Artist & Songs",
            type="primary",
            use_container_width=True
        )
    
    # Process search
    if search_button:
        # Validate inputs
        if not genre:
            st.error("❌ Please enter a music genre!")
            return
        
        if not st.session_state.api_key or not st.session_state.api_key.startswith('sk-'):
            st.error("❌ Please enter a valid OpenAI API key in the sidebar!")
            return
        
        with st.spinner(f"🔍 Searching for {genre} artists with official YouTube channels..."):
            try:
                # Initialize LLM with API key
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    st.error("❌ Failed to initialize OpenAI. Please check your API key.")
                    return
                
                # Create chains
                artist_chain = create_artist_finder_chain(llm)
                song_chain = create_song_finder_chain(llm)
                
                # Create sequential chain
                full_chain = SequentialChain(
                    chains=[artist_chain, song_chain],
                    input_variables=["genre"],
                    output_variables=["artist_info", "songs_info"],
                    verbose=True
                )
                
                # Run chain
                result = full_chain({"genre": genre})
                
                # Debug: Show raw output
                with st.expander("🔍 Debug: Raw LLM Output"):
                    st.write("**Artist Info:**")
                    st.code(result["artist_info"])
                    st.write("**Songs Info:**")
                    st.code(result["songs_info"])
                
                # Parse results
                parsed_result = parse_llm_output(
                    result["artist_info"],
                    result["songs_info"]
                )
                
                if parsed_result:
                    # Store in session state
                    parsed_result["genre"] = genre  # Add genre for reference
                    st.session_state.recommendations.append(parsed_result)
                    
                    # Display results
                    st.success(f"✅ Found **{parsed_result['artist_name']}**!")
                    
                    # Artist info
                    st.markdown(f"### 🎤 {parsed_result['artist_name']}")
                    if parsed_result['artist_description']:
                        st.info(parsed_result['artist_description'])
                    
                    # Songs section
                    st.markdown("### 🎵 Recommended Songs")
                    
                    for idx, song in enumerate(parsed_result['songs'], 1):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{idx}. {song['title']}**")
                            if 'search_query' in song:
                                with st.expander("🔍 Search Query Used"):
                                    st.code(song['search_query'], language=None)
                        
                        with col2:
                            if song['youtube_url']:
                                st.markdown(
                                    f'<a href="{song["youtube_url"]}" target="_blank">'
                                    '<button style="background-color: #FF0000; color: white; '
                                    'border: none; padding: 8px 16px; border-radius: 4px; '
                                    'cursor: pointer;">▶️ Watch</button></a>',
                                    unsafe_allow_html=True
                                )
                        
                        # Display video
                        if song['youtube_url']:
                            try:
                                st.video(song['youtube_url'])
                            except Exception as e:
                                st.warning(f"Could not embed video. Here's the link: {song['youtube_url']}")
                        
                        st.divider()
                    
                else:
                    st.error("❌ Could not parse the results. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    # Display previous recommendations
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Previous Searches")
        
        for i, rec in enumerate(reversed(st.session_state.recommendations)):
            with st.sidebar.expander(f"{rec['artist_name']} ({rec.get('genre', 'Unknown')})"):
                st.write(f"**Artist:** {rec['artist_name']}")
                st.write(f"**Songs found:** {len(rec['songs'])}")
                
                # Add a button to reload this recommendation
                if st.button(f"Load", key=f"load_{i}"):
                    # You could implement loading functionality here
                    st.info("Click the main search button with a new genre to search again!")

if __name__ == "__main__":
    main()
