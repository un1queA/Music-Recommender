# prerequisites:
# pip install langchain-openai youtube-search streamlit httpx
# Run: streamlit run app.py

import streamlit as st
import os
import re
import random
from typing import Dict, Optional, List, Tuple
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

def is_similar_artist(artist1: str, artist2: str) -> bool:
    """Check if two artist names are similar (avoid Stray Kids vs Stray Kids (K-pop))"""
    clean1 = clean_artist_name(artist1)
    clean2 = clean_artist_name(artist2)
    
    # Check exact match
    if clean1 == clean2:
        return True
    
    # Check if one contains the other (common for variations)
    if clean1 in clean2 or clean2 in clean1:
        # But don't match too short names (like "A" in "ABBA")
        if len(clean1) > 3 and len(clean2) > 3:
            return True
    
    return False

# =============================================================================
# PROMPT TEMPLATES (UPDATED FOR BETTER DUPLICATE PREVENTION)
# =============================================================================

def create_artist_prompt(genre: str, excluded_artists: List[str], search_count: int) -> PromptTemplate:
    """Create prompt template for finding artists - adapts based on search count"""
    
    # Determine if we should require official channel
    require_official = search_count < 50  # Require official for first 50 searches
    
    # Prepare exclusion list for prompt (limit to 30 to avoid token limits)
    excluded_text = ""
    if excluded_artists:
        # Use a representative sample, not just the last few
        sample_size = min(30, len(excluded_artists))
        
        # Get a mix: recent + some older ones
        if len(excluded_artists) > sample_size:
            # Take 20 recent and 10 random from older ones
            recent = excluded_artists[-20:]
            older_sample = random.sample(excluded_artists[:-20], min(10, len(excluded_artists)-20))
            sample = list(set(recent + older_sample))[:sample_size]
        else:
            sample = excluded_artists
        
        excluded_text = f"\nDO NOT suggest these artists: {', '.join(sample)}."
    
    # Determine prompt based on search count
    if require_official:
        channel_requirement = "Artist MUST have an official YouTube presence (channel, Vevo, or official topic channel) with official music videos."
    else:
        # After 50 searches, relax requirements
        channel_requirement = "Artist may or may not have an official YouTube channel. It's acceptable if they don't have official videos."
    
    template = f"""You are a music expert. Find ONE artist who specializes in {{genre}} music.
    
IMPORTANT REQUIREMENTS:
1. {channel_requirement}
2. Artist should be well-known or emerging in {{genre}} music
3. Genre can be in any language (K-pop, J-pop, Reggaeton, Hip Hop, Bhangra, etc.)
4. Suggest an artist that hasn't been suggested before for this genre{excluded_text}

Return EXACTLY in this format:
ARTIST: [Artist Name]
DESCRIPTION: [Brief 1-2 sentence description]
HAS_OFFICIAL_CHANNEL: [Yes/No - be honest about this]"""

    return PromptTemplate(input_variables=["genre"], template=template)

def create_song_prompt() -> PromptTemplate:
    """Create prompt template for finding songs from an artist"""
    
    template = """Based on this artist: {artist_info}

Find 3 popular or representative songs by this artist in {genre} style.

IMPORTANT: Provide EXACT song titles as they appear on official releases.

For EACH song, provide:
1. The exact song title
2. A search query that will find the CORRECT video

Format EXACTLY like this:
SONG 1 TITLE: [Title]
SONG 1 SEARCH: [Artist Name] "[Song Title]" official music video

SONG 2 TITLE: [Title]
SONG 2 SEARCH: [Artist Name] "[Song Title]" official

SONG 3 TITLE: [Title]
SONG 3 SEARCH: [Artist Name] "[Song Title]" mv"""

    return PromptTemplate(input_variables=["artist_info", "genre"], template=template)

# =============================================================================
# PARSING FUNCTIONS
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
            parts = line.split(":", 1)
            if len(parts) > 1:
                has_official_channel = parts[1].strip()
    
    return {
        "name": artist_name, 
        "description": artist_desc,
        "has_official_channel": has_official_channel
    }

def extract_songs_info(songs_output: str, artist_name: str) -> List[Dict]:
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
                # Default search with quotes for exact match
                search_query = f'{artist_name} "{title}" official music video'
            
            songs.append({
                "title": title,
                "search_query": search_query,
                "is_official": False
            })
    
    return songs[:3]

# =============================================================================
# IMPROVED YOUTUBE SEARCH WITH STRICT MATCHING
# =============================================================================

def search_youtube_video(search_query: str, artist_name: str, song_title: str, expect_official: bool = True) -> Dict:
    """Search for YouTube video with strict matching to ensure correct song"""
    try:
        # Add quotes for exact phrase search
        if '"' not in search_query:
            search_query = f'"{song_title}" {artist_name}'
        
        results = YoutubeSearch(search_query, max_results=15).to_dict()
        
        clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
        clean_song = re.sub(r'[^\w\s]', '', song_title.lower())
        
        best_match = None
        best_score = -1000  # Start negative to require minimum quality
        best_is_official = False
        best_title = ""
        
        official_indicators = ['official', 'music video', 'mv', '【mv】', 'vevo', 'official video']
        unofficial_indicators = ['cover', 'tribute', 'karaoke', 'fan made', 'lyrics', 'lyric', '1 hour', 'loop']
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Calculate relevance score
            score = 0
            
            # CRITICAL: Song title must be in video title (with some flexibility)
            song_words = set(clean_song.split())
            title_words = set(re.sub(r'[^\w\s]', '', title).split())
            common_words = song_words.intersection(title_words)
            
            if len(song_words) > 0:
                title_match_ratio = len(common_words) / len(song_words)
                if title_match_ratio >= 0.7:  # At least 70% match
                    score += int(100 * title_match_ratio)
                else:
                    # Skip videos that don't match the song title
                    continue
            
            # Artist name in channel (very strong indicator of official)
            if any(word in channel for word in clean_artist.split()):
                score += 60  # Increased weight
            
            # Official indicators
            is_official = any(indicator in title for indicator in official_indicators)
            if is_official:
                score += 40
            
            # Artist name in title
            if any(word in title for word in clean_artist.split()):
                score += 20
            
            # Length check (avoid 1-hour loops, etc.)
            duration = result.get('duration', '')
            if ':' in duration:
                try:
                    mins, secs = map(int, duration.split(':'))
                    total_seconds = mins * 60 + secs
                    # Penalize very long videos (likely loops/compilations)
                    if total_seconds > 600:  # > 10 minutes
                        score -= 30
                except:
                    pass
            
            # Penalize unofficial content if we expect official
            if expect_official:
                if any(indicator in title for indicator in unofficial_indicators):
                    score -= 80  # Heavy penalty for unofficial when expecting official
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = video_url
                best_is_official = is_official
                best_title = result['title']
        
        # Determine if we found a good match
        if best_match and best_score >= 80:  # Higher threshold for quality
            return {
                "url": best_match,
                "is_official": best_is_official,
                "score": best_score,
                "title": best_title,
                "found": True
            }
        elif best_match and best_score >= 40:  # Lower quality but still acceptable
            return {
                "url": best_match,
                "is_official": best_is_official,
                "score": best_score,
                "title": best_title,
                "found": True
            }
        else:
            return {"url": None, "is_official": False, "score": 0, "title": "", "found": False}
            
    except Exception as e:
        st.warning(f"YouTube search error: {str(e)[:100]}")
    
    return {"url": None, "is_official": False, "score": 0, "title": "", "found": False}

# =============================================================================
# MAIN APPLICATION (UPDATED FOR STRICT DUPLICATE PREVENTION)
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
    Discover artists in any music genre. No duplicate artists for the same genre!
    """)
    
    # API Configuration
    st.sidebar.header("🔑 DeepSeek API Configuration")
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
                
                # Get all excluded artists for this genre
                excluded_artists = st.session_state.used_artists_by_genre.get(genre_key, [])
                
                # Try to find a new artist (up to 5 attempts)
                max_attempts = 5
                artist_info = None
                artist_name = ""
                
                for attempt in range(max_attempts):
                    # Create prompt based on search count
                    artist_prompt = create_artist_prompt(genre, excluded_artists, search_count)
                    artist_chain = LLMChain(llm=llm, prompt=artist_prompt)
                    artist_result = artist_chain.run({"genre": genre})
                    
                    # Extract artist info
                    artist_info = extract_artist_info(artist_result)
                    artist_name = artist_info["name"]
                    
                    if not artist_name:
                        continue
                    
                    # Check if this is a duplicate (using improved similarity check)
                    is_duplicate = False
                    for existing_artist in excluded_artists:
                        if is_similar_artist(artist_name, existing_artist):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        break
                    elif attempt < max_attempts - 1:
                        # Add to temporary exclusion for next attempt
                        excluded_artists.append(artist_name)
                
                # If we still got a duplicate after all attempts
                if not artist_name:
                    st.error("❌ Could not find a new artist. Please try a different genre.")
                    return
                
                # Final duplicate check
                is_final_duplicate = False
                for existing_artist in st.session_state.used_artists_by_genre.get(genre_key, []):
                    if is_similar_artist(artist_name, existing_artist):
                        is_final_duplicate = True
                        break
                
                if is_final_duplicate:
                    st.warning(f"⚠️ '{artist_name}' was already suggested for {genre}. Please try searching again!")
                    return
                
                # Determine if we should expect official videos
                has_claimed_official = artist_info.get("has_official_channel", "").lower() == "yes"
                expect_official = has_claimed_official and search_count < 50
                
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
                official_videos_found = 0
                unofficial_videos_found = 0
                no_videos_found = 0
                
                for song in songs:
                    video_result = search_youtube_video(
                        song["search_query"], 
                        artist_name,
                        song["title"],
                        expect_official=expect_official
                    )
                    
                    if video_result["found"]:
                        song["youtube_url"] = video_result["url"]
                        song["is_official"] = video_result["is_official"]
                        song["video_title"] = video_result["title"]
                        song["match_score"] = video_result["score"]
                        
                        if video_result["is_official"]:
                            official_videos_found += 1
                        else:
                            unofficial_videos_found += 1
                    else:
                        song["youtube_url"] = None
                        song["is_official"] = False
                        song["video_title"] = ""
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
                
                # Special message for fallback mode (after 50 searches)
                if search_count >= 50 and not has_claimed_official:
                    st.warning("""
                    ⚠️ **Note**: After 50+ searches for this genre, we're now suggesting artists that may not have official YouTube channels.
                    This is normal when exploring less mainstream artists or niche sub-genres.
                    """)
                
                st.markdown("### 🎵 Recommended Songs")
                
                for idx, song in enumerate(songs, 1):
                    # Create a row for each song
                    col_song_title, col_song_action = st.columns([3, 1])
                    
                    with col_song_title:
                        # Show match quality
                        if song.get("match_score", 0) >= 80:
                            match_text = "🟢 Good match"
                        elif song.get("match_score", 0) >= 40:
                            match_text = "🟡 Fair match"
                        else:
                            match_text = "🔴 Poor match"
                        
                        status_badge = ""
                        if song.get("is_official"):
                            status_badge = " | 🎬 Official"
                        elif song.get("youtube_url"):
                            status_badge = " | 📹 Unofficial"
                        
                        st.markdown(f"**{idx}. {song['title']}** {match_text}{status_badge}")
                        
                        # Show actual video title if different from song title
                        if song.get("video_title") and song["title"].lower() not in song["video_title"].lower():
                            st.caption(f"Video found: *{song['video_title']}*")
                    
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
                            if not has_claimed_official and search_count >= 50:
                                st.caption("""
                                *Why no official video?*
                                - This artist doesn't maintain an official YouTube channel
                                - Their music may be available on other platforms (Spotify, Apple Music, SoundCloud)
                                - They might be an emerging artist without video production
                                - Consider searching on streaming platforms instead
                                """)
                    
                    # Display video if available
                    if song.get('youtube_url'):
                        try:
                            st.video(song['youtube_url'])
                        except Exception:
                            st.markdown(f"[Open video in YouTube]({song['youtube_url']})")
                    
                    st.divider()
                
                # Summary for fallback artists
                if not has_claimed_official and search_count >= 50:
                    st.info("""
                    💡 **About This Artist**: 
                    This artist doesn't have official YouTube videos. To listen to their music:
                    1. Check streaming platforms (Spotify, Apple Music, etc.)
                    2. Search for live performances or interviews
                    3. Look for their music on SoundCloud or Bandcamp
                    4. Check if they have an official website
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
        
        for genre_key, count in list(st.session_state.search_counts_by_genre.items())[-5:]:
            artist_count = len(st.session_state.used_artists_by_genre.get(genre_key, []))
            mode = "Official" if count < 50 else "Fallback"
            st.sidebar.write(f"**{genre_key.title()}**: {artist_count} artists ({mode} mode)")
    
    # Previous Searches
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Previous Searches")
        
        for i, rec in enumerate(reversed(st.session_state.recommendations[-5:])):
            with st.sidebar.expander(f"{rec['artist_name']} ({rec['genre']})"):
                st.write(f"**Search #:** {rec.get('total_searches_for_genre', '?')}")
                st.write(f"**Official Channel:** {'Yes' if rec.get('has_official_channel') else 'No'}")
                official_videos = rec.get('official_videos_found', 0)
                total_songs = len(rec['songs'])
                
                if official_videos == total_songs:
                    st.write("✅ All official videos")
                elif official_videos > 0:
                    st.write(f"🟡 {official_videos}/{total_songs} official")
                else:
                    st.write("🔴 No official videos")

if __name__ == "__main__":
    main()
