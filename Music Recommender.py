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
# PROMPT TEMPLATES (UPDATED FOR ENDLESS SEARCH)
# =============================================================================

def create_artist_prompt(genre: str, excluded_artists: list, attempt_count: int) -> PromptTemplate:
    """Create prompt template for finding artists - adapts based on attempt count"""
    
    excluded_text = ""
    if excluded_artists:
        # Show only last 5 to avoid overwhelming the AI
        excluded_text = f"\nDo not suggest these artists: {', '.join(excluded_artists[-5:])}."
    
    # Tiered approach: First 10 searches require official channels, then relax requirements
    if attempt_count <= 10:
        # STRICT MODE: Official channel required
        channel_requirement = "Artist MUST have an official YouTube presence (channel, Vevo, or official topic channel)"
    else:
        # RELAXED MODE: Artist may not have official channel
        channel_requirement = "Artist may or may not have an official YouTube channel. If they don't, that's okay - just mention they're a great {genre} artist."
    
    template = f"""You are a music expert. Find ONE artist who specializes in {{genre}} music.
    
REQUIREMENTS:
1. {channel_requirement}
2. Artist should be well-known or emerging in {{genre}} music
3. Genre can be in any language (K-pop, J-pop, Reggaeton, Hip Hop, Bhangra, etc.)
4. Provide the exact artist name{excluded_text}

Return EXACTLY in this format:
ARTIST: [Artist Name]
DESCRIPTION: [Brief 1-2 sentence description]
HAS_OFFICIAL_CHANNEL: [Yes/No - based on your knowledge]"""

    return PromptTemplate(input_variables=["genre"], template=template)

def create_song_prompt() -> PromptTemplate:
    """Create prompt template for finding songs from an artist"""
    
    template = """Based on this artist: {artist_info}

Find 3 popular or representative songs by this artist in {genre} style.

For EACH song, provide:
1. The exact song title
2. A search query that might find the video (official or unofficial)

Format EXACTLY like this:
SONG 1 TITLE: [Title]
SONG 1 SEARCH: [Artist Name] [Song Title]

SONG 2 TITLE: [Title]
SONG 2 SEARCH: [Artist Name] [Song Title]

SONG 3 TITLE: [Title]
SONG 3 SEARCH: [Artist Name] [Song Title]"""

    return PromptTemplate(input_variables=["artist_info", "genre"], template=template)

# =============================================================================
# PARSING FUNCTIONS (UPDATED)
# =============================================================================

def extract_artist_info(artist_output: str) -> Dict:
    """Extract artist name and description from LLM output"""
    artist_name = ""
    artist_desc = ""
    has_official_channel = "Unknown"
    
    for line in artist_output.strip().split('\n'):
        line_lower = line.lower()
        if line.startswith("ARTIST:"):
            artist_name = line.replace("ARTIST:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            artist_desc = line.replace("DESCRIPTION:", "").strip()
        elif "has_official_channel:" in line_lower:
            # Extract Yes/No value
            parts = line.split(":", 1)
            if len(parts) > 1:
                has_official_channel = parts[1].strip()
    
    return {
        "name": artist_name, 
        "description": artist_desc,
        "has_official_channel": has_official_channel
    }

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
                search_query = f"{artist_name} {title}"
            
            songs.append({
                "title": title,
                "search_query": search_query,
                "is_official": False  # Default, will update after search
            })
    
    return songs[:3]

# =============================================================================
# YOUTUBE SEARCH (UPDATED)
# =============================================================================

def search_youtube_video(search_query: str, artist_name: str, expect_official: bool = True) -> Dict:
    """Search for YouTube video and return detailed results"""
    try:
        results = YoutubeSearch(search_query, max_results=10).to_dict()
        
        clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
        best_match = None
        best_score = 0
        best_is_official = False
        all_videos = []
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Calculate relevance score
            score = 0
            
            # Artist name in channel (strong indicator)
            if any(word in channel for word in clean_artist.split()):
                score += 40
            
            # Official indicators in title
            official_indicators = ['official', 'music video', 'mv', '【mv】', 'vevo']
            is_official = any(indicator in title for indicator in official_indicators)
            if is_official:
                score += 30
                best_is_official = True
            
            # Artist name in title
            if any(word in title for word in clean_artist.split()):
                score += 20
            
            # Penalize unofficial content if we expect official
            if expect_official:
                unofficial_indicators = ['cover', 'tribute', 'karaoke', 'fan made']
                if any(indicator in title for indicator in unofficial_indicators):
                    score -= 50
            
            # Store video info
            all_videos.append({
                "url": video_url,
                "title": result['title'],
                "channel": result['channel'],
                "score": score,
                "is_official": is_official
            })
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = video_url
        
        # Return detailed results
        if best_match and best_score > 20:  # Lower threshold for relaxed mode
            return {
                "url": best_match,
                "is_official": best_is_official,
                "score": best_score,
                "all_results": all_videos[:3],  # Top 3 alternatives
                "found": True
            }
        elif results:
            # Return first result as fallback
            return {
                "url": f"https://www.youtube.com/watch?v={results[0]['id']}",
                "is_official": any(ind in results[0]['title'].lower() for ind in official_indicators),
                "score": 10,
                "all_results": all_videos[:3],
                "found": True
            }
            
    except Exception as e:
        st.warning(f"YouTube search error: {str(e)[:100]}")
    
    return {"url": None, "is_official": False, "score": 0, "all_results": [], "found": False}

# =============================================================================
# MAIN APPLICATION (UPDATED FOR ENDLESS SEARCH)
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
    if 'search_counts_by_genre' not in st.session_state:
        st.session_state.search_counts_by_genre = {}
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # UI Header
    st.title("🎵 Music Discovery Pro")
    st.markdown("""
    Discover artists in any music genre. Searches never end - we'll always find you someone new!
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
        
        with st.spinner(f"🔍 Finding a unique {genre} artist..."):
            try:
                # Initialize LLM
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    st.error("❌ Failed to initialize DeepSeek. Please check your API key.")
                    return
                
                # Track search count for this genre
                genre_key = genre.lower()
                if genre_key not in st.session_state.search_counts_by_genre:
                    st.session_state.search_counts_by_genre[genre_key] = 0
                
                search_count = st.session_state.search_counts_by_genre[genre_key] + 1
                st.session_state.search_counts_by_genre[genre_key] = search_count
                
                # Get excluded artists
                excluded_artists = st.session_state.used_artists_by_genre.get(genre_key, [])
                
                # Try multiple times if we get a duplicate
                max_attempts = 3
                artist_info = None
                artist_name = ""
                
                for attempt in range(max_attempts):
                    # Chain 1: Find Artist (adapts based on search count)
                    artist_prompt = create_artist_prompt(genre, excluded_artists, search_count)
                    artist_chain = LLMChain(llm=llm, prompt=artist_prompt)
                    artist_result = artist_chain.run({"genre": genre})
                    
                    # Extract artist info
                    artist_info = extract_artist_info(artist_result)
                    artist_name = artist_info["name"]
                    
                    if not artist_name:
                        continue
                    
                    # Check if this is a duplicate
                    clean_new_artist = clean_artist_name(artist_name)
                    existing_clean = [clean_artist_name(a) for a in excluded_artists]
                    
                    if clean_new_artist not in existing_clean:
                        break
                    elif attempt < max_attempts - 1:
                        st.info(f"Found duplicate '{artist_name}'. Trying again...")
                        excluded_artists.append(artist_name)
                
                if not artist_name:
                    st.error("❌ Could not find a suitable artist. Please try again.")
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
                
                # Determine if we should expect official videos
                expect_official = search_count <= 15  # First 15 searches expect official
                has_claimed_official = artist_info.get("has_official_channel", "").lower() == "yes"
                
                # Search for YouTube videos
                official_videos_found = 0
                unofficial_videos_found = 0
                no_videos_found = 0
                
                for song in songs:
                    video_result = search_youtube_video(
                        song["search_query"], 
                        artist_name, 
                        expect_official=expect_official or has_claimed_official
                    )
                    
                    if video_result["found"]:
                        song["youtube_url"] = video_result["url"]
                        song["is_official"] = video_result["is_official"]
                        song["alternative_videos"] = video_result["all_results"]
                        
                        if video_result["is_official"]:
                            official_videos_found += 1
                        else:
                            unofficial_videos_found += 1
                    else:
                        song["youtube_url"] = None
                        song["is_official"] = False
                        no_videos_found += 1
                
                # Store in session state
                recommendation = {
                    "genre": genre,
                    "artist_name": artist_name,
                    "artist_description": artist_info.get("description", ""),
                    "songs": songs,
                    "has_official_channel": has_claimed_official,
                    "official_videos_found": official_videos_found,
                    "total_searches_for_genre": search_count
                }
                st.session_state.recommendations.append(recommendation)
                
                # Track used artist for this genre
                if genre_key not in st.session_state.used_artists_by_genre:
                    st.session_state.used_artists_by_genre[genre_key] = []
                
                if artist_name not in st.session_state.used_artists_by_genre[genre_key]:
                    st.session_state.used_artists_by_genre[genre_key].append(artist_name)
                
                # DISPLAY RESULTS
                st.success(f"✅ Found **{artist_name}**!")
                st.markdown(f"### 🎤 {artist_name}")
                
                if artist_info.get("description"):
                    st.info(artist_info["description"])
                
                # Show search context
                total_suggested = len(st.session_state.used_artists_by_genre[genre_key])
                st.caption(f"*Search #{search_count} for {genre} | Total unique artists: {total_suggested}*")
                
                # Warning if artist doesn't have official channel
                if not has_claimed_official and search_count > 10:
                    st.warning("""
                    ⚠️ **Note**: This artist may not have an official YouTube channel with music videos. 
                    This is common for emerging artists, niche genres, or artists who use different platforms.
                    We'll show available videos, which might be fan uploads or lyric videos.
                    """)
                
                st.markdown("### 🎵 Recommended Songs")
                
                for idx, song in enumerate(songs, 1):
                    # Create a row for each song
                    col_song_title, col_song_action = st.columns([3, 1])
                    
                    with col_song_title:
                        status_badge = ""
                        if song.get("is_official"):
                            status_badge = " 🟢 Official"
                        elif song.get("youtube_url"):
                            status_badge = " 🟡 Unofficial"
                        else:
                            status_badge = " 🔴 Not Found"
                        
                        st.markdown(f"**{idx}. {song['title']}**{status_badge}")
                        
                        if song.get("alternative_videos") and not song.get("is_official"):
                            with st.expander("🔍 Alternative Videos"):
                                for alt in song["alternative_videos"]:
                                    official_tag = " (Official)" if alt["is_official"] else ""
                                    st.write(f"- [{alt['title']}]({alt['url']}){official_tag}")
                    
                    with col_song_action:
                        if song.get('youtube_url'):
                            button_color = "#FF0000" if song.get("is_official") else "#FF9900"
                            button_text = "▶️ Official" if song.get("is_official") else "▶️ Watch"
                            
                            st.markdown(
                                f'<a href="{song["youtube_url"]}" target="_blank">'
                                f'<button style="background-color: {button_color}; color: white; '
                                'border: none; padding: 8px 16px; border-radius: 4px; '
                                'cursor: pointer; margin-top: 10px;">{button_text}</button></a>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("❌ No video found")
                            st.caption("""
                            *Why no video?*
                            - Artist may not have official YouTube presence
                            - Videos might be region-restricted
                            - Content may be on other platforms (Spotify, Apple Music)
                            """)
                    
                    # Display video if available
                    if song.get('youtube_url'):
                        try:
                            st.video(song['youtube_url'])
                        except Exception:
                            st.markdown(f"[Open video in YouTube]({song['youtube_url']})")
                    
                    st.divider()
                
                # Final summary
                if official_videos_found == 0 and unofficial_videos_found > 0:
                    st.info("""
                    📝 **About These Videos**: 
                    The videos shown are unofficial uploads (covers, lyric videos, fan uploads). 
                    To support this artist, consider streaming their music on platforms like Spotify, Apple Music, or their official website.
                    """)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # Session Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Session Tools")
    
    if st.sidebar.button("🗑️ Clear All History", type="secondary"):
        st.session_state.used_artists_by_genre = {}
        st.session_state.search_counts_by_genre = {}
        st.session_state.recommendations = []
        st.sidebar.success("✅ All history cleared!")
        st.rerun()
    
    # Statistics Display
    if st.session_state.search_counts_by_genre:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Search Statistics")
        
        for genre_key, count in st.session_state.search_counts_by_genre.items():
            artist_count = len(st.session_state.used_artists_by_genre.get(genre_key, []))
            st.sidebar.write(f"**{genre_key.title()}**: {count} searches, {artist_count} unique artists")
    
    # Previous Searches
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Previous Searches")
        
        for i, rec in enumerate(reversed(st.session_state.recommendations[-5:])):
            with st.sidebar.expander(f"{rec['artist_name']} ({rec['genre']})"):
                st.write(f"**Genre:** {rec['genre']}")
                st.write(f"**Artist:** {rec['artist_name']}")
                official_videos = rec.get('official_videos_found', 0)
                total_songs = len(rec['songs'])
                
                if official_videos == total_songs:
                    st.write("✅ All official videos found")
                elif official_videos > 0:
                    st.write(f"🟡 {official_videos}/{total_songs} official videos")
                else:
                    st.write("🔴 No official videos found")
    
    # App Info
    with st.sidebar.expander("📊 About This App"):
        st.markdown("""
        **How it works:**
        1. Enter any music genre (English or non-English)
        2. AI finds an artist in that genre
        3. First 15 searches prioritize artists with official YouTube channels
        4. After that, we suggest any artist in the genre
        5. App searches YouTube for available videos
        
        **Video Availability:**
        - 🟢 Official: From artist's official channel
        - 🟡 Unofficial: Fan uploads, lyric videos, covers
        - 🔴 Not Found: No YouTube videos available
        
        **Why some artists lack videos:**
        - Emerging artists without YouTube presence
        - Regional restrictions or licensing issues
        - Artists using other platforms (Spotify, SoundCloud)
        - Niche genres with limited online presence
        """)

if __name__ == "__main__":
    main()
