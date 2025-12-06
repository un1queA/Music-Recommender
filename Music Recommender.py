# prerequisites:
# pip install langchain-openai youtube-search streamlit httpx
# Run: streamlit run app.py

import streamlit as st
import os
import re
from typing import List, Dict, Optional, Tuple
from youtube_search import YoutubeSearch

# Import LangChain components - updated for DeepSeek
from langchain_openai import ChatOpenAI  # For the DeepSeek API (OpenAI-compatible)
from langchain_core.prompts import PromptTemplate  # Prompts moved here
from langchain.chains import LLMChain, SequentialChain  # Legacy chains moved here

# =============================================================================
# DEEPSEEK API CONFIGURATION
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        # DeepSeek uses OpenAI-compatible API format
        llm = ChatOpenAI(
            model="deepseek-chat",  # Use "deepseek-chat" for latest model
            temperature=0.8,  # Increased for more variety
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",  # DeepSeek API endpoint
            max_tokens=1500,
            timeout=30  # Add timeout for reliability
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

# =============================================================================
# ARTIST NAME CLEANING AND DUPLICATE PREVENTION
# =============================================================================

def clean_artist_name(artist: str) -> str:
    """Clean artist name to avoid duplicates like 'lisa' and 'lisa (japanese singer)'"""
    # Remove anything in parentheses and extra spaces
    clean_name = re.sub(r'\([^)]*\)', '', artist).strip()
    # Remove special characters and convert to lowercase
    clean_name = re.sub(r'[^\w\s]', '', clean_name).lower()
    return clean_name

# =============================================================================
# PROMPT TEMPLATES (Optimized for multiple artists)
# =============================================================================

def create_artist_finder_chain(llm: ChatOpenAI, excluded_artists: List[str] = None):
    """Create chain to find multiple artists with official YouTube channels"""
    
    excluded_text = ""
    if excluded_artists:
        excluded_text = f" Avoid these artists that were already suggested: {', '.join(excluded_artists[:5])}."
    
    artist_prompt = PromptTemplate(
        input_variables=["genre"],
        template=f"""
        You are a music expert. Find 3 DIFFERENT artists who specialize in {{genre}} music and have official YouTube channels.
        
        IMPORTANT REQUIREMENTS:
        1. Each artist MUST have an official YouTube presence (channel or Vevo)
        2. All artists should be known for {{genre}} music
        3. Genre can be in any language
        4. Provide accurate information
        5. Artists should be diverse (different sub-styles, regions, or eras within {genre})
        {excluded_text}
        
        Return EXACTLY in this format for EACH artist:
        ARTIST: [Artist Name]
        DESCRIPTION: [Brief description - 1-2 sentences]
        ---
        """
    )
    
    return LLMChain(llm=llm, prompt=artist_prompt, output_key="artists_info")

def create_song_finder_chain(llm: ChatOpenAI):
    """Create chain to find songs from official channel"""
    
    song_prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""
        Based on the artist: {artist_name}
        
        Suggest 3 popular songs by this artist in {genre} style that likely have official YouTube videos.
        
        For EACH song, provide:
        1. Exact song title
        2. Search query to find the video
        
        Format EXACTLY like this:
        SONG 1 TITLE: [Title]
        SONG 1 SEARCH: {artist_name} [Title] official music video
        
        SONG 2 TITLE: [Title]
        SONG 2 SEARCH: {artist_name} [Title] official video
        
        SONG 3 TITLE: [Title]
        SONG 3 SEARCH: {artist_name} [Title] official
        """
    )
    
    return LLMChain(llm=llm, prompt=song_prompt, output_key="songs_info")

# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_artists_from_output(artists_output: str) -> List[Dict]:
    """Parse multiple artists from LLM output"""
    artists = []
    current_artist = {}
    
    for line in artists_output.strip().split('\n'):
        line = line.strip()
        if line.startswith("ARTIST:"):
            if current_artist:  # Save previous artist if exists
                artists.append(current_artist)
            current_artist = {"name": line.replace("ARTIST:", "").strip(), "description": ""}
        elif line.startswith("DESCRIPTION:") and current_artist:
            current_artist["description"] = line.replace("DESCRIPTION:", "").strip()
        elif line == "---" and current_artist:  # Separator line
            artists.append(current_artist)
            current_artist = {}
    
    # Don't forget the last artist
    if current_artist:
        artists.append(current_artist)
    
    return artists[:3]  # Return max 3 artists

def parse_songs_from_output(songs_output: str, artist_name: str) -> List[Dict]:
    """Parse songs from LLM output"""
    songs = []
    lines = songs_output.strip().split('\n')
    
    for i in range(0, len(lines), 3):
        if i < len(lines) and "TITLE:" in lines[i]:
            title = lines[i].split("TITLE:", 1)[1].strip()
            search_query = ""
            
            if i + 1 < len(lines) and "SEARCH:" in lines[i + 1]:
                search_query = lines[i + 1].split("SEARCH:", 1)[1].strip()
            else:
                search_query = f"{artist_name} {title} official music video"
            
            songs.append({
                "title": title,
                "search_query": search_query
            })
    
    return songs[:3]

# =============================================================================
# YOUTUBE SEARCH FUNCTIONS (Enhanced)
# =============================================================================

def search_youtube_video(search_query: str, artist_name: str) -> Optional[str]:
    """Search for official YouTube video with enhanced filtering"""
    try:
        results = YoutubeSearch(search_query, max_results=10).to_dict()
        
        # Clean artist name for comparison
        clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
        
        best_match = None
        best_score = 0
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            
            # Calculate match score
            score = 0
            
            # Check if channel contains artist name
            if any(word in channel for word in clean_artist.split()):
                score += 40
            
            # Official indicators
            official_indicators = ['official', 'music video', 'mv', '【mv】']
            if any(indicator in title for indicator in official_indicators):
                score += 30
            
            # Artist in title
            if any(word in title for word in clean_artist.split()):
                score += 20
            
            # Avoid unofficial content
            unofficial_indicators = ['cover', 'tribute', 'karaoke', 'instrumental', 'remix']
            if any(indicator in title for indicator in unofficial_indicators):
                score -= 50
            
            # Track best match
            if score > best_score:
                best_score = score
                best_match = video_id
        
        if best_match and best_score > 30:
            return f"https://www.youtube.com/watch?v={best_match}"
        elif results:
            # Fallback to first result
            return f"https://www.youtube.com/watch?v={results[0]['id']}"
            
    except Exception as e:
        st.warning(f"YouTube search error: {str(e)[:100]}")
    
    return None

# =============================================================================
# MAIN APPLICATION (Updated for multi-artist support)
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
    if 'used_artists' not in st.session_state:
        st.session_state.used_artists = set()
    if 'available_artists' not in st.session_state:
        st.session_state.available_artists = []
    if 'selected_artist_index' not in st.session_state:
        st.session_state.selected_artist_index = 0
    
    # Title and description
    st.title("🎵 Music Discovery Pro")
    st.markdown("""
    Discover multiple artists with official YouTube channels in any music genre.
    Powered by **DeepSeek AI**.
    """)
    
    # DeepSeek API Key input
    st.sidebar.header("🔑 DeepSeek API Configuration")
    
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
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    with col1:
        genre = st.text_input(
            "Enter a music genre:",
            placeholder="e.g., K-pop, Lo-fi Hip Hop, Flamenco, Vaporwave, Bhangra...",
            key="genre_input"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        search_button = st.button(
            "🔍 Find Artists",
            type="primary",
            use_container_width=True
        )
    
    # Process genre search
    if search_button and genre:
        if not st.session_state.api_key:
            st.error("❌ Please enter your DeepSeek API key in the sidebar!")
            return
        
        with st.spinner(f"🔍 Finding unique {genre} artists..."):
            try:
                # Initialize LLM
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    st.error("❌ Failed to initialize DeepSeek. Please check your API key.")
                    return
                
                # Get already used artists for this genre
                excluded_artists = []
                for rec in st.session_state.recommendations:
                    if rec.get("genre", "").lower() == genre.lower():
                        excluded_artists.append(rec["artist_name"])
                
                # Create artist finder chain
                artist_chain = create_artist_finder_chain(llm, excluded_artists)
                
                # Get multiple artists
                result = artist_chain.run({"genre": genre})
                artists = parse_artists_from_output(result)
                
                # Filter out already used artists
                filtered_artists = []
                for artist in artists:
                    clean_name = clean_artist_name(artist["name"])
                    if clean_name not in st.session_state.used_artists:
                        filtered_artists.append(artist)
                
                if not filtered_artists:
                    st.warning("⚠️ Couldn't find new artists for this genre. Try a different genre or reset the session.")
                    return
                
                # Store available artists
                st.session_state.available_artists = filtered_artists
                st.session_state.selected_artist_index = 0
                
                # Display available artists
                st.success(f"✅ Found {len(filtered_artists)} unique {genre} artists!")
                st.markdown("---")
                
                # Show artist selection
                for idx, artist in enumerate(filtered_artists):
                    with st.container():
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            if st.button(f"Select #{idx+1}", key=f"select_{idx}"):
                                st.session_state.selected_artist_index = idx
                                st.rerun()
                        with col_b:
                            st.markdown(f"**{artist['name']}**")
                            if artist['description']:
                                st.caption(artist['description'])
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error finding artists: {str(e)}")
    
    # Display songs for selected artist
    if st.session_state.available_artists and genre:
        selected_idx = st.session_state.selected_artist_index
        if selected_idx < len(st.session_state.available_artists):
            selected_artist = st.session_state.available_artists[selected_idx]
            
            st.markdown(f"### 🎤 Selected: **{selected_artist['name']}**")
            if selected_artist['description']:
                st.info(selected_artist['description'])
            
            if st.button("🎵 Get Songs for This Artist", type="secondary"):
                with st.spinner(f"Finding songs for {selected_artist['name']}..."):
                    try:
                        # Initialize LLM
                        llm = initialize_llm(st.session_state.api_key)
                        if not llm:
                            return
                        
                        # Create song finder chain
                        song_chain = create_song_finder_chain(llm)
                        songs_result = song_chain.run({
                            "artist_name": selected_artist["name"],
                            "genre": genre
                        })
                        
                        songs = parse_songs_from_output(songs_result, selected_artist["name"])
                        
                        # Search for YouTube videos
                        for idx, song in enumerate(songs, 1):
                            youtube_url = search_youtube_video(song["search_query"], selected_artist["name"])
                            song["youtube_url"] = youtube_url
                        
                        # Store in recommendations
                        recommendation = {
                            "genre": genre,
                            "artist_name": selected_artist["name"],
                            "artist_description": selected_artist["description"],
                            "songs": songs
                        }
                        st.session_state.recommendations.append(recommendation)
                        
                        # Mark artist as used
                        clean_name = clean_artist_name(selected_artist["name"])
                        st.session_state.used_artists.add(clean_name)
                        
                        # Display results
                        st.markdown("### 🎵 Recommended Songs")
                        
                        for idx, song in enumerate(songs, 1):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {song['title']}**")
                                with st.expander("🔍 Search Query"):
                                    st.code(song['search_query'], language=None)
                            
                            with col2:
                                if song.get('youtube_url'):
                                    st.markdown(
                                        f'<a href="{song["youtube_url"]}" target="_blank">'
                                        '<button style="background-color: #FF0000; color: white; '
                                        'border: none; padding: 8px 16px; border-radius: 4px; '
                                        'cursor: pointer;">▶️ Watch</button></a>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.warning("Video not found")
                            
                            if song.get('youtube_url'):
                                try:
                                    st.video(song['youtube_url'])
                                except:
                                    st.markdown(f"[Open in YouTube]({song['youtube_url']})")
                            
                            st.divider()
                        
                    except Exception as e:
                        st.error(f"❌ Error getting songs: {str(e)}")
    
    # Session management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Session Tools")
    
    if st.sidebar.button("🔄 Reset Used Artists", type="secondary"):
        st.session_state.used_artists = set()
        st.session_state.available_artists = []
        st.success("Artist history cleared!")
        st.rerun()
    
    # Display previous recommendations
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Previous Searches")
        
        for i, rec in enumerate(reversed(st.session_state.recommendations[-10:])):
            with st.sidebar.expander(f"{rec['artist_name']} ({rec.get('genre', 'Unknown')})"):
                st.write(f"**Genre:** {rec['genre']}")
                st.write(f"**Artist:** {rec['artist_name']}")
                st.write(f"**Songs found:** {len(rec['songs'])}")
    
    # DeepSeek info
    with st.sidebar.expander("📊 About DeepSeek"):
        st.markdown("""
        **Tips for best results:**
        1. Be specific with genres (e.g., "Japanese City Pop" vs "City Pop")
        2. The app now suggests multiple artists per search
        3. Click "Reset Used Artists" to clear the duplicate filter
        4. Each artist can be selected individually for song recommendations
        """)

if __name__ == "__main__":
    main()
