# prerequisites:
# pip install langchain-openai youtube-search streamlit httpx
# Run: streamlit run app.py

import streamlit as st
import os
import re
from typing import Dict, Optional
from youtube_search import YoutubeSearch

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# =============================================================================
# DEEPSEEK API & INITIALIZATION
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.8,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1000,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def clean_artist_name(artist: str) -> str:
    """Clean artist name to avoid duplicates"""
    clean_name = re.sub(r'\([^)]*\)', '', artist).strip()
    clean_name = re.sub(r'[^\w\s]', '', clean_name).lower()
    return clean_name

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def create_artist_prompt(genre: str, excluded_artists: list) -> PromptTemplate:
    """Create prompt template for finding artists with official YouTube channels"""
    
    excluded_text = ""
    if excluded_artists:
        excluded_text = f"\nCRITICAL: DO NOT suggest these artists (already used for '{genre}'): {', '.join(excluded_artists[:3])}. You MUST suggest a DIFFERENT artist."
    
    template = f"""You are a music expert. Find ONE artist who specializes in {{genre}} music and has an official YouTube channel.

REQUIREMENTS:
1. Artist MUST have an official YouTube presence (channel, Vevo, or official topic channel)
2. Artist should be well-known for {{genre}} music
3. Genre can be in any language (K-pop, J-pop, Reggaeton, Hip Hop, Bhangra, etc.)
4. Provide the exact artist name as it appears on YouTube{excluded_text}

Return EXACTLY in this format:
ARTIST: [Artist Name]
DESCRIPTION: [Brief 1-2 sentence description]"""
    
    return PromptTemplate(input_variables=["genre"], template=template)

def create_song_prompt() -> PromptTemplate:
    """Create prompt template for finding songs from an artist's channel"""
    
    template = """Based on this artist: {artist_info}

Find 3 popular songs from their OFFICIAL YouTube channel that represent {genre} music.

For EACH song, provide:
1. The exact song title as it appears on YouTube
2. A search query that will find the official video

Format EXACTLY like this:
SONG 1 TITLE: [Title]
SONG 1 SEARCH: [Artist Name] [Song Title] official music video

SONG 2 TITLE: [Title]
SONG 2 SEARCH: [Artist Name] [Song Title] official video

SONG 3 TITLE: [Title]
SONG 3 SEARCH: [Artist Name] [Song Title] official"""
    
    return PromptTemplate(input_variables=["artist_info", "genre"], template=template)

# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def extract_artist_info(artist_output: str) -> Dict:
    """Extract artist name and description from LLM output"""
    artist_name = ""
    artist_desc = ""
    
    for line in artist_output.strip().split('\n'):
        if line.startswith("ARTIST:"):
            artist_name = line.replace("ARTIST:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            artist_desc = line.replace("DESCRIPTION:", "").strip()
    
    return {"name": artist_name, "description": artist_desc}

def extract_songs_info(songs_output: str, artist_name: str) -> list:
    """Extract song information from LLM output"""
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
    
    return songs[:3]  # Return max 3 songs

# =============================================================================
# YOUTUBE SEARCH
# =============================================================================

def search_youtube_video(search_query: str, artist_name: str) -> Optional[str]:
    """Search for official YouTube video with smart filtering"""
    try:
        results = YoutubeSearch(search_query, max_results=10).to_dict()
        
        clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
        best_match = None
        best_score = 0
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            
            # Calculate relevance score
            score = 0
            
            # Strong indicator: artist name in channel
            if any(word in channel for word in clean_artist.split()):
                score += 40
            
            # Official indicators in title
            if any(indicator in title for indicator in ['official', 'music video', 'mv', '【mv】']):
                score += 30
            
            # Artist name in title
            if any(word in title for word in clean_artist.split()):
                score += 20
            
            # Penalize unofficial content
            if any(indicator in title for indicator in ['cover', 'tribute', 'karaoke', 'fan made']):
                score -= 50
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = video_id
        
        # Return result if it meets threshold, otherwise first result as fallback
        if best_match and best_score > 30:
            return f"https://www.youtube.com/watch?v={best_match}"
        elif results:
            return f"https://www.youtube.com/watch?v={results[0]['id']}"
            
    except Exception as e:
        st.warning(f"YouTube search error: {str(e)[:100]}")
    
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
    if 'used_artists_by_genre' not in st.session_state:
        st.session_state.used_artists_by_genre = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # UI Header
    st.title("🎵 Music Discovery Pro")
    st.markdown("""
    Discover artists with official YouTube channels in any music genre.
    *No duplicate artists for the same genre within this session!*
    """)
    
    # API Configuration
    st.sidebar.header("🔑 DeepSeek API Configuration")
    with st.sidebar.expander("ℹ️ Get API Key"):
        st.markdown("""
        1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
        2. Sign up or log in
        3. Go to "API Keys" section
        4. Create and copy your API key
        5. Free tier available with generous limits
        """)
    
    api_key = st.sidebar.text_input(
        "DeepSeek API Key:",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Enter your DeepSeek API key",
        key="api_key_input"
    )
    
    if api_key:
        st.session_state.api_key = api_key
    
    # Genre Input
    genre = st.text_input(
        "Enter a music genre:",
        placeholder="e.g., K-pop, Lo-fi Hip Hop, Reggaeton, Bhangra, Synthwave...",
        key="genre_input"
    )
    
    # Search Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button(
            "🎯 Find Artist & Songs",
            type="primary",
            use_container_width=True
        )
    
    # Process Search
    if search_button:
        if not genre:
            st.error("❌ Please enter a music genre!")
            return
        
        if not st.session_state.api_key:
            st.error("❌ Please enter your DeepSeek API key in the sidebar!")
            return
        
        with st.spinner(f"🔍 Searching for a unique {genre} artist..."):
            try:
                # Initialize LLM
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    st.error("❌ Failed to initialize DeepSeek. Please check your API key.")
                    return
                
                # Check for duplicates
                genre_key = genre.lower()
                excluded_artists = st.session_state.used_artists_by_genre.get(genre_key, [])
                
                # Chain 1: Find Artist
                artist_prompt = create_artist_prompt(genre, excluded_artists)
                artist_chain = LLMChain(llm=llm, prompt=artist_prompt)
                artist_result = artist_chain.run({"genre": genre})
                
                # Extract artist info
                artist_info = extract_artist_info(artist_result)
                artist_name = artist_info["name"]
                
                if not artist_name:
                    st.error("❌ Could not find a suitable artist. Please try again.")
                    return
                
                # Check for duplicates (case-insensitive)
                clean_new_artist = clean_artist_name(artist_name)
                existing_clean = [clean_artist_name(a) for a in excluded_artists]
                
                if clean_new_artist in existing_clean:
                    st.warning(f"⚠️ '{artist_name}' was already suggested for {genre}. Please try searching again!")
                    return
                
                # Chain 2: Find Songs
                song_prompt = create_song_prompt()
                song_chain = LLMChain(llm=llm, prompt=song_prompt)
                songs_result = song_chain.run({
                    "artist_info": f"{artist_name} - {artist_info.get('description', '')}",
                    "genre": genre
                })
                
                # Extract songs
                songs = extract_songs_info(songs_result, artist_name)
                
                # Search for YouTube videos
                for song in songs:
                    song["youtube_url"] = search_youtube_video(song["search_query"], artist_name)
                
                # Store in session state
                recommendation = {
                    "genre": genre,
                    "artist_name": artist_name,
                    "artist_description": artist_info.get("description", ""),
                    "songs": songs
                }
                st.session_state.recommendations.append(recommendation)
                
                # Track used artist for this genre
                if genre_key not in st.session_state.used_artists_by_genre:
                    st.session_state.used_artists_by_genre[genre_key] = []
                st.session_state.used_artists_by_genre[genre_key].append(artist_name)
                
                # Display Results
                st.success(f"✅ Found **{artist_name}**!")
                st.markdown(f"### 🎤 {artist_name}")
                
                if artist_info.get("description"):
                    st.info(artist_info["description"])
                
                st.markdown("### 🎵 Recommended Songs")
                
                for idx, song in enumerate(songs, 1):
                    # Create a row for each song
                    col_song_title, col_song_action = st.columns([3, 1])
                    
                    with col_song_title:
                        st.markdown(f"**{idx}. {song['title']}**")
                        with st.expander("🔍 Search Query Used"):
                            st.code(song['search_query'], language="text")
                    
                    with col_song_action:
                        if song.get('youtube_url'):
                            st.markdown(
                                f'<a href="{song["youtube_url"]}" target="_blank">'
                                '<button style="background-color: #FF0000; color: white; '
                                'border: none; padding: 8px 16px; border-radius: 4px; '
                                'cursor: pointer; margin-top: 10px;">▶️ Watch</button></a>',
                                unsafe_allow_html=True
                            )
                    
                    # Display video
                    if song.get('youtube_url'):
                        try:
                            st.video(song['youtube_url'])
                        except Exception:
                            st.markdown(f"[Open video in YouTube]({song['youtube_url']})")
                    
                    st.divider()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # Session Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Session Tools")
    
    if st.sidebar.button("🗑️ Clear History for Current Genre", type="secondary"):
        if genre:
            genre_key = genre.lower()
            if genre_key in st.session_state.used_artists_by_genre:
                st.session_state.used_artists_by_genre[genre_key] = []
                st.sidebar.success(f"✅ Cleared artist history for '{genre}'!")
        else:
            st.sidebar.warning("⚠️ Enter a genre first to clear its history.")
    
    if st.sidebar.button("🗑️ Clear All Genre History", type="secondary"):
        st.session_state.used_artists_by_genre = {}
        st.sidebar.success("✅ Cleared all genre history!")
    
    # Previous Searches
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Previous Searches")
        
        for i, rec in enumerate(reversed(st.session_state.recommendations[-5:])):
            with st.sidebar.expander(f"{rec['artist_name']} ({rec['genre']})"):
                st.write(f"**Genre:** {rec['genre']}")
                st.write(f"**Artist:** {rec['artist_name']}")
                if rec.get('artist_description'):
                    st.write(f"**Description:** {rec['artist_description']}")
                st.write(f"**Songs found:** {len(rec['songs'])}")
                
                for j, song in enumerate(rec['songs'][:2], 1):
                    st.write(f"{j}. {song['title']}")
                if len(rec['songs']) > 2:
                    st.write(f"3. {rec['songs'][2]['title']}")
    
    # App Info
    with st.sidebar.expander("📊 About This App"):
        st.markdown("""
        **How it works:**
        1. Enter any music genre (English or non-English)
        2. AI finds an artist with an official YouTube channel
        3. AI suggests 3 songs from that channel
        4. App searches YouTube for official videos
        5. Watch videos directly in the app
        
        **Duplicate Prevention:**
        - Same genre → Different artist each time
        - Artists tracked per genre
        - Use the clear buttons to reset history
        
        **Tips:**
        - Be specific (e.g., "Japanese City Pop" not just "Pop")
        - Try genres in different languages
        """)

if __name__ == "__main__":
    main()
