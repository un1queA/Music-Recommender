# prerequisites:
# pip install langchain-openai youtube-search streamlit httpx
# Run: streamlit run app.py

import streamlit as st
import os
import re
from typing import List, Dict, Optional
from youtube_search import YoutubeSearch

# Import LangChain components - updated for DeepSeek
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =============================================================================
# DEEPSEEK API CONFIGURATION
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        # DeepSeek uses OpenAI-compatible API format
        llm = ChatOpenAI(
            model="deepseek-chat",  # Use "deepseek-chat" for latest model
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",  # DeepSeek API endpoint
            max_tokens=1000,
            timeout=30  # Add timeout for reliability
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

# =============================================================================
# PROMPT TEMPLATES (Optimized for DeepSeek)
# =============================================================================

def create_artist_finder_chain(llm: ChatOpenAI):
    """Create chain to find artist with official YouTube channel"""
    
    artist_prompt = PromptTemplate(
        input_variables=["genre"],
        template="""
        You are a music expert. Find an artist who specializes in {genre} music and has an official YouTube channel.
        
        IMPORTANT:
        1. The artist MUST have an official YouTube presence (channel or Vevo)
        2. Artist should be known for {genre} music
        3. Genre can be in any language
        4. Provide accurate information
        
        Return EXACTLY in this format:
        ARTIST: [Artist Name]
        DESCRIPTION: [Brief description]
        """
    )
    
    return LLMChain(llm=llm, prompt=artist_prompt, output_key="artist_info")

def create_song_finder_chain(llm: ChatOpenAI):
    """Create chain to find songs from official channel"""
    
    song_prompt = PromptTemplate(
        input_variables=["artist_info", "genre"],
        template="""
        Based on: {artist_info}
        
        Suggest 3 popular songs by this artist in {genre} style that likely have official YouTube videos.
        
        For EACH song, provide:
        1. Exact song title
        2. Search query to find the video
        
        Format EXACTLY like this:
        SONG 1 TITLE: [Title]
        SONG 1 SEARCH: [Artist] [Title] official
        
        SONG 2 TITLE: [Title]
        SONG 2 SEARCH: [Artist] [Title] official music video
        
        SONG 3 TITLE: [Title]
        SONG 3 SEARCH: [Artist] [Title] official video
        """
    )
    
    return LLMChain(llm=llm, prompt=song_prompt, output_key="songs_info")

# =============================================================================
# YOUTUBE SEARCH FUNCTIONS (No changes needed)
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
# MAIN APPLICATION (Updated for DeepSeek)
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
    Powered by **DeepSeek AI**.
    """)
    
    # DeepSeek API Key input
    st.sidebar.header("🔑 DeepSeek API Configuration")
    
    # Instructions for getting DeepSeek API key
    with st.sidebar.expander("ℹ️ Get DeepSeek API Key"):
        st.markdown("""
        1. Go to [DeepSeek Platform](https://platform.deepseek.com/)
        2. Sign up or log in
        3. Navigate to "API Keys"
        4. Create a new API key
        5. Free tier available with generous limits
        """)
    
    api_key = st.sidebar.text_input(
        "DeepSeek API Key:",
        type="password",
        value=st.session_state.api_key,
        placeholder="Enter your DeepSeek API key...",
        help="Get your API key from https://platform.deepseek.com",
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
        
        if not st.session_state.api_key:
            st.error("❌ Please enter your DeepSeek API key in the sidebar!")
            return
        
        with st.spinner(f"🔍 Searching for {genre} artists with DeepSeek AI..."):
            try:
                # Initialize LLM with API key
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    st.error("❌ Failed to initialize DeepSeek. Please check your API key.")
                    return
                
                # Create chains
                artist_chain = create_artist_finder_chain(llm)
                song_chain = create_song_finder_chain(llm)
                
                # Create sequential chain
                full_chain = SequentialChain(
                    chains=[artist_chain, song_chain],
                    input_variables=["genre"],
                    output_variables=["artist_info", "songs_info"],
                    verbose=False  # Set to True for debugging
                )
                
                # Run chain
                result = full_chain({"genre": genre})
                
                # Debug: Show raw output (optional)
                with st.expander("🔍 Debug: Raw AI Output"):
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
                    parsed_result["genre"] = genre
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
                    st.info("Click the main search button with a new genre to search again!")
    
    # Add DeepSeek specific info
    with st.sidebar.expander("📊 About DeepSeek"):
        st.markdown("""
        **DeepSeek AI Features:**
        - Free tier with generous limits
        - OpenAI-compatible API
        - Good music knowledge
        - Fast response times
        
        **Tips for best results:**
        1. Be specific with genres
        2. Try both English and non-English genres
        3. For niche genres, add descriptors
           (e.g., "Japanese City Pop" instead of just "City Pop")
        """)

if __name__ == "__main__":
    main()


