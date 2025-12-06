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
from langchain.chains import LLMChain, SequentialChain

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
# LANGCHAIN CHAINS & PROMPTS
# =============================================================================

def create_artist_finder_chain(llm: ChatOpenAI, genre: str):
    """Create chain to find a UNIQUE artist for the given genre."""
    
    # Get already used artists for this genre
    genre_key = genre.lower()
    excluded_artists = st.session_state.used_artists_by_genre.get(genre_key, [])
    
    excluded_text = ""
    if excluded_artists:
        excluded_text = f"\nIMPORTANT: DO NOT suggest these artists: {', '.join(excluded_artists[:5])}. Suggest a DIFFERENT artist."
    
    artist_prompt = PromptTemplate(
        input_variables=["genre"],
        template=f"""You are a music expert. Find ONE artist who specializes in {{genre}} music and has an official YouTube channel.
        
REQUIREMENTS:
1. Artist MUST have official YouTube presence (channel, Vevo, or topic channel)
2. Artist should be well-known for {{genre}} music
3. Genre can be in any language (K-pop, Reggaeton, Hip Hop, etc.)
4. Provide exact artist name as on YouTube{excluded_text}

Return EXACTLY:
ARTIST: [Artist Name]
DESCRIPTION: [Brief 1-sentence description]"""
    )
    
    return LLMChain(llm=llm, prompt=artist_prompt, output_key="artist_info")

def create_song_finder_chain(llm: ChatOpenAI):
    """Create chain to find songs from official channel"""
    
    song_prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""Based on artist: {artist_name}

Suggest 3 popular songs by this artist in {genre} style that have official YouTube videos.

For EACH song:
1. Exact song title
2. Search query to find video

Format EXACTLY:
SONG 1 TITLE: [Title]
SONG 1 SEARCH: {artist_name} [Title] official music video

SONG 2 TITLE: [Title]
SONG 2 SEARCH: {artist_name} [Title] official video

SONG 3 TITLE: [Title]
SONG 3 SEARCH: {artist_name} [Title] official"""
    )
    
    return LLMChain(llm=llm, prompt=song_prompt, output_key="songs_info")

# =============================================================================
# PARSING & YOUTUBE SEARCH
# =============================================================================

def parse_llm_output(artist_output: str, songs_output: str) -> Dict:
    """Parse LLM outputs into structured data"""
    try:
        # Parse artist
        artist_name = ""
        artist_desc = ""
        
        for line in artist_output.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist_name = line.replace("ARTIST:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                artist_desc = line.replace("DESCRIPTION:", "").strip()
        
        # Parse songs
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
                
                songs.append({"title": title, "search_query": search_query})
        
        return {
            "artist_name": artist_name,
            "artist_description": artist_desc,
            "songs": songs[:3]
        }
        
    except Exception as e:
        st.error(f"Error parsing output: {e}")
        return None

def search_youtube_video(search_query: str, artist_name: str) -> Optional[str]:
    """Search for official YouTube video"""
    try:
        results = YoutubeSearch(search_query, max_results=10).to_dict()
        
        clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
        best_match = None
        best_score = 0
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            
            score = 0
            if any(word in channel for word in clean_artist.split()):
                score += 40
            if any(indicator in title for indicator in ['official', 'music video', 'mv']):
                score += 30
            if any(word in title for word in clean_artist.split()):
                score += 20
            if any(indicator in title for indicator in ['cover', 'tribute', 'karaoke']):
                score -= 50
            
            if score > best_score:
                best_score = score
                best_match = video_id
        
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
    st.set_page_config(page_title="🎵 Music Discovery Pro", page_icon="🎵", layout="wide")
    
    # Initialize session state (CRITICAL - fixes your error)
    if 'used_artists_by_genre' not in st.session_state:
        st.session_state.used_artists_by_genre = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # UI
    st.title("🎵 Music Discovery Pro")
    st.markdown("Find artists with official YouTube channels. No duplicates for the same genre!")
    
    # API Key
    st.sidebar.header("🔑 DeepSeek API")
    api_key = st.sidebar.text_input(
        "API Key:", type="password", value=st.session_state.api_key,
        placeholder="Enter DeepSeek API key...", key="api_key_input"
    )
    if api_key:
        st.session_state.api_key = api_key
    
    # Genre input
    genre = st.text_input(
        "Enter music genre:",
        placeholder="e.g., K-pop, Lo-fi Hip Hop, Reggaeton, Bhangra...",
        key="genre_input"
    )
    
    search_button = st.button("🎯 Find Artist & Songs", type="primary", use_container_width=True)
    
    # Process search
    if search_button:
        if not genre:
            st.error("❌ Please enter a genre!")
            return
        if not st.session_state.api_key:
            st.error("❌ Please enter your API key!")
            return
        
        with st.spinner(f"🔍 Finding unique {genre} artist..."):
            try:
                # Initialize LLM
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    return
                
                # Create and run chains
                artist_chain = create_artist_finder_chain(llm, genre)
                song_chain = create_song_finder_chain(llm)
                
                full_chain = SequentialChain(
                    chains=[artist_chain, song_chain],
                    input_variables=["genre"],
                    output_variables=["artist_info", "songs_info"],
                    verbose=False
                )
                
                result = full_chain({"genre": genre})
                
                # Parse and validate result
                parsed_result = parse_llm_output(result["artist_info"], result["songs_info"])
                
                if not parsed_result or not parsed_result["artist_name"]:
                    st.error("❌ Could not find artist. Please try again.")
                    return
                
                # TRACK DUPLICATES
                genre_key = genre.lower()
                artist_clean = clean_artist_name(parsed_result["artist_name"])
                
                # Check if artist already used for this genre
                if genre_key in st.session_state.used_artists_by_genre:
                    if artist_clean in [clean_artist_name(a) for a in st.session_state.used_artists_by_genre[genre_key]]:
                        st.warning(f"⚠️ '{parsed_result['artist_name']}' was already suggested for {genre}. Try searching again for a different artist!")
                        return
                
                # Store the new artist
                if genre_key not in st.session_state.used_artists_by_genre:
                    st.session_state.used_artists_by_genre[genre_key] = []
                st.session_state.used_artists_by_genre[genre_key].append(parsed_result["artist_name"])
                
                # Search for YouTube videos
                for song in parsed_result["songs"]:
                    song["youtube_url"] = search_youtube_video(song["search_query"], parsed_result["artist_name"])
                
                # Store recommendation
                parsed_result["genre"] = genre
                st.session_state.recommendations.append(parsed_result)
                
                # DISPLAY RESULTS
                st.success(f"✅ Found **{parsed_result['artist_name']}**!")
                st.markdown(f"### 🎤 {parsed_result['artist_name']}")
                if parsed_result['artist_description']:
                    st.info(parsed_result['artist_description'])
                
                st.markdown("### 🎵 Recommended Songs")
                for idx, song in enumerate(parsed_result['songs'], 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{idx}. {song['title']}**")
                    with col2:
                        if song.get('youtube_url'):
                            st.markdown(f'<a href="{song["youtube_url"]}" target="_blank"><button style="background-color: #FF0000; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">▶️ Watch</button></a>', unsafe_allow_html=True)
                    
                    if song.get('youtube_url'):
                        try:
                            st.video(song['youtube_url'])
                        except:
                            st.markdown(f"[Open in YouTube]({song['youtube_url']})")
                    st.divider()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Session management sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Session Tools")
    
    if st.sidebar.button("🗑️ Clear History for Current Genre", key="clear_genre"):
        if genre:
            genre_key = genre.lower()
            if genre_key in st.session_state.used_artists_by_genre:
                st.session_state.used_artists_by_genre[genre_key] = []
                st.sidebar.success(f"Cleared history for '{genre}'!")
        else:
            st.sidebar.warning("Enter a genre first.")
    
    if st.sidebar.button("🗑️ Clear All Genre History", key="clear_all"):
        st.session_state.used_artists_by_genre = {}
        st.sidebar.success("Cleared all history!")
    
    # Previous searches
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Previous Searches")
        for rec in reversed(st.session_state.recommendations[-5:]):
            with st.sidebar.expander(f"{rec['artist_name']} ({rec['genre']})"):
                st.write(f"**Artist:** {rec['artist_name']}")
                st.write(f"**Songs:** {len(rec['songs'])}")

if __name__ == "__main__":
    main()
