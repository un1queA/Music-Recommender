# prerequisites:
# pip install langchain-openai youtube-search streamlit httpx
# Run: streamlit run app.py

import streamlit as st
import os
import re
import random
import time
from typing import Dict, Optional, List, Tuple
from youtube_search import YoutubeSearch
from urllib.parse import quote_plus

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
            temperature=0.7,  # Slightly lower for more consistent channel URLs
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1000,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

# =============================================================================
# YOUTUBE CHANNEL DISCOVERY
# =============================================================================

def find_youtube_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Dict:
    """Use DeepSeek to find the artist's official YouTube channel URL"""
    
    channel_prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""Find the OFFICIAL YouTube channel for the artist: {artist_name}
        
Artist specializes in {genre} music.

IMPORTANT: Provide the EXACT YouTube channel URL if known. If not certain, say "UNKNOWN".

Search the web knowledge to find:
1. Official YouTube channel URL (like https://www.youtube.com/c/ChannelName or https://www.youtube.com/user/Username)
2. Or YouTube music topic channel (like https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxx)
3. Or VEVO channel if applicable

Return EXACTLY in this format:
CHANNEL_URL: [YouTube channel URL or UNKNOWN]
CHANNEL_NAME: [Channel name as it appears on YouTube]
CONFIDENCE: [High/Medium/Low based on certainty]"""
    )
    
    channel_chain = LLMChain(llm=llm, prompt=channel_prompt)
    
    try:
        result = channel_chain.run({"artist_name": artist_name, "genre": genre})
        
        # Parse the result
        channel_url = "UNKNOWN"
        channel_name = ""
        confidence = "Low"
        
        for line in result.strip().split('\n'):
            if line.startswith("CHANNEL_URL:"):
                channel_url = line.replace("CHANNEL_URL:", "").strip()
            elif line.startswith("CHANNEL_NAME:"):
                channel_name = line.replace("CHANNEL_NAME:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip()
        
        return {
            "url": channel_url if channel_url != "UNKNOWN" else None,
            "name": channel_name,
            "confidence": confidence,
            "found": channel_url != "UNKNOWN"
        }
        
    except Exception as e:
        st.warning(f"Error finding YouTube channel: {e}")
        return {"url": None, "name": "", "confidence": "Low", "found": False}

def extract_channel_id_from_url(channel_url: str) -> Optional[str]:
    """Extract YouTube channel ID from various URL formats"""
    try:
        # Handle different YouTube URL formats
        patterns = [
            r'youtube\.com/channel/([^/?&]+)',  # /channel/UC...
            r'youtube\.com/c/([^/?&]+)',        # /c/ChannelName
            r'youtube\.com/user/([^/?&]+)',     # /user/Username
            r'youtube\.com/@([^/?&]+)',         # /@Handle
        ]
        
        for pattern in patterns:
            match = re.search(pattern, channel_url, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    except:
        return None

# =============================================================================
# CHANNEL-LOCKED YOUTUBE SEARCH
# =============================================================================

def search_within_channel(song_title: str, artist_name: str, channel_info: Dict) -> Dict:
    """Search for a video specifically within an artist's YouTube channel"""
    
    if not channel_info.get("url"):
        return {"found": False, "reason": "No channel URL available"}
    
    try:
        # Create search query that includes channel name
        search_query = f'{song_title} {artist_name} {channel_info["name"]}'
        
        # Search YouTube
        results = YoutubeSearch(search_query, max_results=20).to_dict()
        
        if not results:
            return {"found": False, "reason": "No search results"}
        
        # Extract channel ID/name patterns from the found channel URL
        target_channel_patterns = []
        channel_url = channel_info["url"].lower()
        
        # Try to extract identifying parts of the channel URL
        if "/channel/" in channel_url:
            channel_id = extract_channel_id_from_url(channel_url)
            if channel_id:
                target_channel_patterns.append(channel_id.lower())
        
        if "/c/" in channel_url:
            target_channel_patterns.append(channel_url.split("/c/")[1].split("/")[0].lower())
        
        if "/user/" in channel_url:
            target_channel_patterns.append(channel_url.split("/user/")[1].split("/")[0].lower())
        
        if "/@" in channel_url:
            target_channel_patterns.append(channel_url.split("/@")[1].split("/")[0].lower())
        
        # Also use the channel name we have
        if channel_info["name"]:
            target_channel_patterns.append(channel_info["name"].lower())
        
        # Search for videos from the target channel
        best_video = None
        best_score = 0
        
        clean_song = re.sub(r'[^\w\s]', '', song_title.lower())
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            
            # Check if this video is from our target channel
            is_target_channel = False
            for pattern in target_channel_patterns:
                if pattern and pattern in channel:
                    is_target_channel = True
                    break
            
            # Skip videos not from target channel
            if not is_target_channel:
                continue
            
            # Calculate relevance score
            score = 0
            
            # Song title matching (most important)
            song_words = set(clean_song.split())
            title_words = set(re.sub(r'[^\w\s]', '', title).split())
            common_words = song_words.intersection(title_words)
            
            if len(song_words) > 0:
                title_match_ratio = len(common_words) / len(song_words)
                score += int(100 * title_match_ratio)
            
            # Official indicators
            official_indicators = ['official', 'music video', 'mv', '【mv】', 'vevo']
            if any(indicator in title for indicator in official_indicators):
                score += 50
            
            # Artist name in title
            clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
            if any(word in title for word in clean_artist.split()):
                score += 30
            
            # Update best match
            if score > best_score:
                best_score = score
                best_video = {
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": result['title'],
                    "channel": result['channel'],
                    "score": score,
                    "is_official": any(ind in title for ind in official_indicators),
                    "search_method": "channel_locked"
                }
        
        if best_video and best_score >= 40:
            best_video["found"] = True
            return best_video
        else:
            return {
                "found": False, 
                "reason": "No matching videos found in channel",
                "search_method": "channel_locked"
            }
            
    except Exception as e:
        return {"found": False, "reason": f"Search error: {str(e)}", "search_method": "channel_locked"}

def fallback_youtube_search(song_title: str, artist_name: str) -> Dict:
    """Fallback search when channel search fails"""
    try:
        search_query = f'{artist_name} "{song_title}" official music video'
        results = YoutubeSearch(search_query, max_results=10).to_dict()
        
        if not results:
            return {"found": False, "reason": "No videos found"}
        
        clean_artist = re.sub(r'[^\w\s]', '', artist_name.lower())
        clean_song = re.sub(r'[^\w\s]', '', song_title.lower())
        
        best_video = None
        best_score = 0
        
        for result in results:
            title = result['title'].lower()
            channel = result['channel'].lower()
            video_id = result['id']
            
            score = 0
            
            # Artist in channel (strong indicator)
            if any(word in channel for word in clean_artist.split()):
                score += 60
            
            # Song title matching
            song_words = set(clean_song.split())
            title_words = set(re.sub(r'[^\w\s]', '', title).split())
            common_words = song_words.intersection(title_words)
            
            if len(song_words) > 0:
                title_match_ratio = len(common_words) / len(song_words)
                score += int(80 * title_match_ratio)
            
            # Official indicators
            if any(indicator in title for indicator in ['official', 'music video', 'mv']):
                score += 40
            
            if score > best_score:
                best_score = score
                best_video = {
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": result['title'],
                    "channel": result['channel'],
                    "score": score,
                    "is_official": any(ind in title for ind in ['official', 'music video', 'mv']),
                    "search_method": "fallback"
                }
        
        if best_video and best_score >= 50:
            best_video["found"] = True
            return best_video
        else:
            return {"found": False, "reason": "No suitable videos found", "search_method": "fallback"}
            
    except Exception as e:
        return {"found": False, "reason": f"Search error: {str(e)}", "search_method": "fallback"}

# =============================================================================
# UPDATED ARTIST PROMPT (INCLUDES CHANNEL REQUIREMENT)
# =============================================================================

def create_artist_prompt(genre: str, excluded_artists: List[str], search_count: int) -> PromptTemplate:
    """Create prompt template for finding artists with known YouTube channels"""
    
    excluded_text = ""
    if excluded_artists:
        sample_size = min(20, len(excluded_artists))
        sample = excluded_artists[-sample_size:] if len(excluded_artists) > sample_size else excluded_artists
        excluded_text = f"\nDO NOT suggest these artists: {', '.join(sample)}."
    
    # Different requirements based on search count
    if search_count < 30:
        channel_req = "Artist MUST have a known, identifiable official YouTube channel with music videos."
    elif search_count < 50:
        channel_req = "Artist should preferably have a YouTube channel, but it's acceptable if not easily found."
    else:
        channel_req = "Artist may not have an official YouTube channel. If they don't, that's acceptable."
    
    template = f"""Find ONE {genre} artist.

REQUIREMENTS:
1. {channel_req}
2. The artist should be clearly associated with {genre} music
3. If the artist has a YouTube channel, it should be findable/searchable{excluded_text}

Return EXACTLY:
ARTIST: [Artist Name]
DESCRIPTION: [Brief description]
KNOWN_FOR_YOUTUBE: [Yes/No - if artist is known to have YouTube content]"""

    return PromptTemplate(input_variables=["genre"], template=template)

# =============================================================================
# MAIN APPLICATION WITH CHANNEL-LOCKED SEARCH
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
    if 'artist_channels' not in st.session_state:
        st.session_state.artist_channels = {}  # Store found channels
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # UI Header
    st.title("🎵 Music Discovery Pro")
    st.markdown("""
    Discover artists with **channel-locked video search** - only showing videos from official channels!
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
        placeholder="e.g., K-pop, Lo-fi Hip Hop, Reggaeton, Bhangra...",
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
            st.error("❌ Please enter your DeepSeek API key!")
            return
        
        with st.spinner(f"🔍 Finding {genre} artist with YouTube channel..."):
            try:
                # Initialize LLM
                llm = initialize_llm(st.session_state.api_key)
                if not llm:
                    st.error("❌ Failed to initialize DeepSeek. Please check your API key.")
                    return
                
                # Track search count
                genre_key = genre.lower()
                if genre_key not in st.session_state.search_counts_by_genre:
                    st.session_state.search_counts_by_genre[genre_key] = 0
                
                search_count = st.session_state.search_counts_by_genre[genre_key] + 1
                st.session_state.search_counts_by_genre[genre_key] = search_count
                
                # Get excluded artists
                excluded_artists = st.session_state.used_artists_by_genre.get(genre_key, [])
                
                # Find artist
                artist_prompt = create_artist_prompt(genre, excluded_artists, search_count)
                artist_chain = LLMChain(llm=llm, prompt=artist_prompt)
                artist_result = artist_chain.run({"genre": genre})
                
                # Parse artist info
                artist_name = ""
                artist_desc = ""
                known_for_youtube = "No"
                
                for line in artist_result.strip().split('\n'):
                    if line.startswith("ARTIST:"):
                        artist_name = line.replace("ARTIST:", "").strip()
                    elif line.startswith("DESCRIPTION:"):
                        artist_desc = line.replace("DESCRIPTION:", "").strip()
                    elif line.startswith("KNOWN_FOR_YOUTUBE:"):
                        known_for_youtube = line.replace("KNOWN_FOR_YOUTUBE:", "").strip()
                
                if not artist_name:
                    st.error("❌ Could not find artist. Please try again.")
                    return
                
                # Check for duplicates
                clean_new = re.sub(r'[^\w\s]', '', artist_name.lower())
                for existing in excluded_artists:
                    clean_existing = re.sub(r'[^\w\s]', '', existing.lower())
                    if clean_new == clean_existing:
                        st.warning(f"⚠️ '{artist_name}' was already suggested. Please try again!")
                        return
                
                # STEP 1: FIND YOUTUBE CHANNEL
                with st.spinner(f"🔍 Searching for {artist_name}'s YouTube channel..."):
                    channel_info = find_youtube_channel(artist_name, genre, llm)
                
                channel_locked = channel_info["found"]
                
                # STEP 2: GET SONGS
                song_prompt = PromptTemplate(
                    input_variables=["artist_name", "genre"],
                    template="""Suggest 3 songs by {artist_name} in {genre} style.

Return EXACTLY:
SONG 1: [Song Title 1]
SONG 2: [Song Title 2]
SONG 3: [Song Title 3]"""
                )
                
                song_chain = LLMChain(llm=llm, prompt=song_prompt)
                songs_result = song_chain.run({"artist_name": artist_name, "genre": genre})
                
                # Parse songs
                songs = []
                for line in songs_result.strip().split('\n'):
                    if ":" in line:
                        song_title = line.split(":", 1)[1].strip()
                        if song_title:
                            songs.append({"title": song_title})
                
                songs = songs[:3]
                
                # STEP 3: SEARCH FOR VIDEOS (CHANNEL-LOCKED OR FALLBACK)
                video_results = []
                channel_videos_found = 0
                fallback_videos_found = 0
                
                for song in songs:
                    video_result = None
                    
                    # Try channel-locked search first if we have a channel
                    if channel_locked:
                        with st.spinner(f"Searching {channel_info['name']} for '{song['title']}'..."):
                            video_result = search_within_channel(
                                song["title"], 
                                artist_name, 
                                channel_info
                            )
                        
                        if video_result.get("found"):
                            channel_videos_found += 1
                    
                    # Fallback if channel search failed
                    if not video_result or not video_result.get("found"):
                        with st.spinner(f"Broad search for '{song['title']}'..."):
                            video_result = fallback_youtube_search(song["title"], artist_name)
                        
                        if video_result.get("found"):
                            fallback_videos_found += 1
                    
                    # Store result
                    song.update(video_result)
                    video_results.append(video_result)
                
                # Store everything
                if genre_key not in st.session_state.used_artists_by_genre:
                    st.session_state.used_artists_by_genre[genre_key] = []
                st.session_state.used_artists_by_genre[genre_key].append(artist_name)
                
                if channel_locked:
                    st.session_state.artist_channels[artist_name] = channel_info
                
                recommendation = {
                    "genre": genre,
                    "artist_name": artist_name,
                    "artist_description": artist_desc,
                    "channel_locked": channel_locked,
                    "channel_info": channel_info if channel_locked else None,
                    "songs": songs,
                    "search_count": search_count,
                    "channel_videos": channel_videos_found,
                    "fallback_videos": fallback_videos_found
                }
                st.session_state.recommendations.append(recommendation)
                
                # DISPLAY RESULTS
                st.success(f"✅ Found **{artist_name}**!")
                
                # Show channel info if found
                if channel_locked:
                    st.info(f"**YouTube Channel Found:** [{channel_info['name']}]({channel_info['url']}) (Confidence: {channel_info['confidence']})")
                    if channel_videos_found > 0:
                        st.success(f"🔒 Found {channel_videos_found} video(s) directly from artist's channel!")
                else:
                    st.warning("⚠️ Could not identify official YouTube channel. Using broader search.")
                
                st.markdown(f"### 🎤 {artist_name}")
                if artist_desc:
                    st.write(artist_desc)
                
                st.markdown("### 🎵 Recommended Songs")
                
                for idx, song in enumerate(songs, 1):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        # Show search method indicator
                        if song.get("search_method") == "channel_locked":
                            badge = "🔒 Channel-Locked"
                            color = "green"
                        elif song.get("search_method") == "fallback":
                            badge = "🌐 Broad Search"
                            color = "orange"
                        else:
                            badge = "❓ Unknown"
                            color = "gray"
                        
                        st.markdown(f"**{idx}. {song['title']}**")
                        st.markdown(f'<span style="color: {color}; font-size: 0.8em;">{badge}</span>', unsafe_allow_html=True)
                        
                        if song.get("channel"):
                            st.caption(f"Channel: {song['channel']}")
                    
                    with col2:
                        if song.get("found"):
                            st.markdown(
                                f'<a href="{song["url"]}" target="_blank">'
                                '<button style="background-color: #FF0000; color: white; '
                                'border: none; padding: 8px 16px; border-radius: 4px; '
                                'cursor: pointer;">▶️ Watch</button></a>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("❌ Not found")
                    
                    with col3:
                        if song.get("is_official"):
                            st.success("Official")
                        elif song.get("found"):
                            st.warning("Unofficial")
                    
                    # Display video if found
                    if song.get("found"):
                        try:
                            st.video(song["url"])
                        except:
                            st.markdown(f"[Open in YouTube]({song['url']})")
                    else:
                        st.info("No suitable video found. Try searching on streaming platforms.")
                    
                    st.divider()
                
                # Summary
                if channel_locked and channel_videos_found < len(songs):
                    st.info(f"""
                    **Search Summary:** 
                    - {channel_videos_found} video(s) found directly from artist's channel
                    - {fallback_videos_found} video(s) found through broader search
                    - Videos marked 🔒 are guaranteed from the artist's official channel
                    """)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Session Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Session Tools")
    
    if st.sidebar.button("🗑️ Clear History", type="secondary"):
        st.session_state.used_artists_by_genre = {}
        st.session_state.search_counts_by_genre = {}
        st.session_state.artist_channels = {}
        st.session_state.recommendations = []
        st.sidebar.success("✅ Cleared!")
        st.rerun()
    
    # Statistics
    if st.session_state.search_counts_by_genre:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Statistics")
        
        for genre_key, count in list(st.session_state.search_counts_by_genre.items())[-3:]:
            artists = st.session_state.used_artists_by_genre.get(genre_key, [])
            channel_count = sum(1 for a in artists if a in st.session_state.artist_channels)
            
            st.sidebar.write(f"**{genre_key.title()}**:")
            st.sidebar.write(f"  {len(artists)} artists")
            st.sidebar.write(f"  {channel_count} with channels found")
    
    # Previous Searches
    if st.session_state.recommendations:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 Recent")
        
        for rec in reversed(st.session_state.recommendations[-3:]):
            with st.sidebar.expander(f"{rec['artist_name'][:15]}..."):
                st.write(f"**Channel:** {'🔒 Found' if rec['channel_locked'] else '🌐 Not found'}")
                st.write(f"**Videos:** {rec.get('channel_videos', 0)}🔒 + {rec.get('fallback_videos', 0)}🌐")

if __name__ == "__main__":
    main()
