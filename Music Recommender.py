# prerequisites:
# pip install langchain-openai youtube-search streamlit langchain_core
# Run: streamlit run app.py

import streamlit as st
import re
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
from urllib.parse import quote_plus
import time
import hashlib

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =============================================================================
# PHASE 1: FIND OFFICIAL CHANNEL FIRST (MOST IMPORTANT)
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1000,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def find_official_youtube_channel(artist_name: str) -> Tuple[Optional[str], str, float]:
    """
    Phase 1: Find the OFFICIAL YouTube channel FIRST with high confidence
    Returns: (channel_url, channel_name, confidence)
    """
    
    # Strategy 1: Search for artist with "official" and "channel"
    primary_search = f"{artist_name} official channel"
    secondary_search = f"{artist_name} topic"
    
    best_channel = None
    best_channel_name = ""
    best_confidence = 0.0
    
    for search_query in [primary_search, secondary_search]:
        try:
            results = YoutubeSearch(search_query, max_results=15).to_dict()
            
            for result in results:
                channel = result['channel']
                title = result['title'].lower()
                channel_lower = channel.lower()
                artist_lower = artist_name.lower()
                
                # Calculate channel confidence score
                score = 0.0
                
                # 1. Artist name in channel (CRITICAL)
                if artist_lower in channel_lower:
                    score += 40
                
                # 2. "Official" in channel name (STRONG SIGNAL)
                if 'official' in channel_lower:
                    score += 35
                
                # 3. "Topic" in channel (YouTube Music official channel)
                if 'topic' in channel_lower:
                    score += 30
                
                # 4. "VEVO" in channel (major label)
                if 'vevo' in channel_lower:
                    score += 25
                
                # 5. "Music" in channel name
                if 'music' in channel_lower:
                    score += 15
                
                # 6. Check if this is actually a channel result (not just a video)
                if any(term in title for term in ['channel', 'subscribe', 'youtube.com']):
                    score += 10
                
                # 7. Penalize unrelated terms
                if any(term in channel_lower for term in ['cover', 'tribute', 'fan', 'reaction']):
                    score -= 20
                
                # Update best channel
                if score > best_confidence and score >= 50:  # Minimum threshold
                    best_confidence = score
                    best_channel_name = channel
                    # Construct approximate channel URL
                    channel_slug = channel.lower().replace(' ', '')
                    best_channel = f"https://www.youtube.com/c/{channel_slug}"
        
        except Exception as e:
            continue
    
    return best_channel, best_channel_name, min(best_confidence / 100.0, 1.0)

def verify_and_lock_channel(artist_name: str, llm: ChatOpenAI, excluded_channels: List[str]) -> Dict:
    """
    Use LLM to verify the found channel and get exact channel details
    """
    
    # First, find channel via YouTube search
    channel_url, channel_name, confidence = find_official_youtube_channel(artist_name)
    
    if not channel_name:
        # Fallback: Ask LLM for channel info
        channel_prompt = PromptTemplate(
            input_variables=["artist_name", "excluded_channels"],
            template="""Find the OFFICIAL YouTube channel for: {artist_name}
            
            The channel MUST:
            1. Be the artist's official music channel (not fan channel, not cover channel)
            2. Have the verified music note badge (♪) on YouTube
            3. Contain official music videos, not just live performances
            
            Already excluded channels: {excluded_channels}
            
            Return EXACTLY:
            CHANNEL_NAME: [Exact channel name as it appears on YouTube]
            CHANNEL_TYPE: [VEVO/Topic/OfficialArtist/MusicLabel/Independent]
            SEARCH_TIP: [Best search term to find this channel on YouTube]"""
        )
        
        channel_chain = LLMChain(llm=llm, prompt=channel_prompt)
        try:
            result = channel_chain.run({
                "artist_name": artist_name,
                "excluded_channels": ", ".join(excluded_channels) if excluded_channels else "None"
            })
            
            # Parse result
            for line in result.strip().split('\n'):
                if line.startswith("CHANNEL_NAME:"):
                    channel_name = line.replace("CHANNEL_NAME:", "").strip()
                elif line.startswith("SEARCH_TIP:"):
                    search_tip = line.replace("SEARCH_TIP:", "").strip()
                    # Try searching with this tip
                    new_url, new_name, new_conf = find_official_youtube_channel(search_tip)
                    if new_name and new_conf > confidence:
                        channel_url, channel_name, confidence = new_url, new_name, new_conf
        
        except Exception as e:
            st.warning(f"LLM channel verification failed: {e}")
    
    return {
        "url": channel_url,
        "name": channel_name,
        "confidence": confidence,
        "found": bool(channel_name and confidence > 0.5)
    }

# =============================================================================
# PHASE 2: CHANNEL-LOCKED SONG SEARCH
# =============================================================================

def search_within_channel_only(song_title: str, artist_name: str, channel_name: str, max_attempts: int = 5) -> Dict:
    """
    Phase 2: Search ONLY within the locked channel
    Returns video from the specified channel or nothing
    """
    
    if not channel_name:
        return {"found": False, "error": "No channel specified"}
    
    # Create search variations prioritizing exact matches
    search_variations = [
        f"{song_title} {artist_name} {channel_name}",
        f'"{song_title}" "{channel_name}"',
        f"{artist_name} {song_title} official {channel_name}",
        f"{song_title} {channel_name}",
        f"{song_title} official music video {channel_name}"
    ]
    
    for search_query in search_variations:
        try:
            results = YoutubeSearch(search_query, max_results=15).to_dict()
            
            # Filter to ONLY videos from our locked channel
            target_channel_lower = channel_name.lower()
            channel_videos = []
            
            for result in results:
                if result['channel'].lower() == target_channel_lower:
                    channel_videos.append(result)
            
            # If we found videos from our channel, pick the best one
            if channel_videos:
                best_video = None
                best_score = 0
                
                for video in channel_videos:
                    score = 0
                    title = video['title'].lower()
                    clean_song = re.sub(r'[^\w\s]', '', song_title.lower())
                    
                    # Title contains exact song title
                    if clean_song in re.sub(r'[^\w\s]', '', title):
                        score += 50
                    
                    # Title contains major words from song title
                    song_words = set(clean_song.split())
                    title_words = set(re.sub(r'[^\w\s]', '', title).split())
                    common_words = song_words.intersection(title_words)
                    if common_words:
                        score += len(common_words) * 10
                    
                    # Official indicators
                    if any(indicator in title for indicator in ['official', 'music video', 'mv', '【official】']):
                        score += 30
                    
                    # Not a live version
                    if not any(term in title for term in ['live', 'concert', 'performance']):
                        score += 15
                    
                    # Good duration (typical song length)
                    if video.get('duration'):
                        try:
                            time_parts = video['duration'].split(':')
                            if len(time_parts) == 2:
                                minutes = int(time_parts[0])
                                if 2 <= minutes <= 7:
                                    score += 10
                        except:
                            pass
                    
                    if score > best_score:
                        best_score = score
                        best_video = video
                
                if best_video and best_score >= 40:  # Good match threshold
                    return {
                        "found": True,
                        "url": f"https://www.youtube.com/watch?v={best_video['id']}",
                        "title": best_video['title'],
                        "channel": best_video['channel'],
                        "score": best_score,
                        "locked_channel": True,
                        "search_query": search_query,
                        "note": f"✅ Found in locked channel: {channel_name}"
                    }
        
        except Exception as e:
            continue
    
    # No suitable video found in the locked channel
    return {
        "found": False,
        "error": f"No video found in channel: {channel_name}",
        "suggestion": f"Try manual search: '{song_title} site:youtube.com/c/{channel_name.lower().replace(' ', '')}'",
        "locked_channel": True
    }

# =============================================================================
# ARTIST DISCOVERY WITH CHANNEL-FIRST APPROACH
# =============================================================================

def discover_artist_with_channel(genre: str, llm: ChatOpenAI, excluded_artists: List[str]) -> Dict:
    """
    Find artist for genre with channel-first approach
    """
    
    excluded_text = "\n".join([f"- {artist}" for artist in excluded_artists[:5]])
    if excluded_artists and len(excluded_artists) > 5:
        excluded_text += f"\n- ... and {len(excluded_artists) - 5} more"
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text"],
        template="""Find an artist for the music genre: {genre}
        
        CRITICAL REQUIREMENTS:
        1. The artist MUST have an OFFICIAL YouTube channel with the verified music note badge (♪)
        2. The channel should have multiple official music videos
        3. The channel name should clearly match the artist name
        
        Already excluded artists:
        {excluded_text}
        
        Return EXACTLY:
        ARTIST: [Artist Name]
        WHY_CHANNEL_EXISTS: [Why this artist likely has a good official YouTube channel]
        EXPECTED_CHANNEL_NAME: [What their YouTube channel name probably is]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "genre": genre,
            "excluded_text": excluded_text if excluded_artists else "None"
        })
        
        artist = ""
        channel_reason = ""
        expected_channel = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist = line.replace("ARTIST:", "").strip()
            elif line.startswith("WHY_CHANNEL_EXISTS:"):
                channel_reason = line.replace("WHY_CHANNEL_EXISTS:", "").strip()
            elif line.startswith("EXPECTED_CHANNEL_NAME:"):
                expected_channel = line.replace("EXPECTED_CHANNEL_NAME:", "").strip()
        
        return {
            "artist": artist,
            "channel_reason": channel_reason,
            "expected_channel": expected_channel,
            "success": bool(artist)
        }
    
    except Exception as e:
        st.warning(f"Artist discovery failed: {e}")
        return {"success": False}

def get_songs_for_artist(artist: str, genre: str, llm: ChatOpenAI) -> List[Dict]:
    """Get 3 songs that likely have official videos"""
    
    prompt = PromptTemplate(
        input_variables=["artist", "genre"],
        template="""For the artist {artist} in genre {genre}, suggest 3 songs that:
        
        1. Are their most popular hits (will definitely have official videos)
        2. Have clear, searchable titles
        3. Are not remixes or live versions (original studio versions)
        
        Return EXACTLY:
        SONG_1: [Song Title 1]
        SONG_2: [Song Title 2]
        SONG_3: [Song Title 3]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist": artist, "genre": genre})
        songs = []
        
        for line in result.strip().split('\n'):
            if ":" in line and ("SONG" in line or "Song" in line):
                song_title = line.split(":", 1)[1].strip()
                if song_title:
                    songs.append({"title": song_title})
        
        return songs[:3]
    
    except Exception as e:
        # Fallback: generic popular songs pattern
        return [
            {"title": f"{artist}'s most popular song"},
            {"title": f"{artist}'s latest hit"},
            {"title": f"{artist}'s signature song"}
        ]

# =============================================================================
# STREAMLIT APP WITH CHANNEL-LOCKING ARCHITECTURE
# =============================================================================

def main():
    st.set_page_config(
        page_title="🔒 Channel-Locked Music Discovery",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize session state
    if 'discovery_sessions' not in st.session_state:
        st.session_state.discovery_sessions = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'locked_channels' not in st.session_state:
        st.session_state.locked_channels = {}  # artist -> channel info
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    
    # Custom CSS
    st.markdown("""
    <style>
    .channel-locked {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
    }
    .channel-failed {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .video-found {
        border: 2px solid #4caf50;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .video-notfound {
        border: 2px solid #ff9800;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔒 Channel-Locked Music Discovery")
    st.markdown("""
    **New Architecture:**
    1. **First find the official YouTube channel** (with verification)
    2. **Lock all song searches to that channel only**
    3. **Never show videos from wrong channels**
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input(
            "DeepSeek API Key:",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Required for artist discovery"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        st.write(f"Total sessions: {len(st.session_state.discovery_sessions)}")
        st.write(f"Locked channels: {len(st.session_state.locked_channels)}")
        
        if st.session_state.locked_channels:
            st.markdown("#### 🔒 Currently Locked Channels")
            for artist, channel in list(st.session_state.locked_channels.items())[-5:]:
                with st.expander(f"{artist[:15]}...", expanded=False):
                    st.write(f"Channel: {channel.get('name', 'Unknown')[:20]}")
                    st.write(f"Confidence: {channel.get('confidence', 0)*100:.0f}%")
        
        if st.button("🔄 Clear All Sessions", type="secondary"):
            st.session_state.discovery_sessions = []
            st.session_state.locked_channels = {}
            st.session_state.excluded_artists = []
            st.success("All sessions cleared!")
            st.rerun()
    
    # Main input
    st.markdown("### 🎯 Enter Music Genre")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        genre_input = st.text_input(
            "Genre:",
            placeholder="e.g., synthwave, K-pop, reggae, classical, hip-hop...",
            label_visibility="collapsed",
            key="genre_input"
        )
    
    with col2:
        st.write("")
        search_btn = st.button(
            "🚀 Find Artist & Channel",
            type="primary",
            use_container_width=True
        )
    
    # Process search
    if search_btn and genre_input:
        if not st.session_state.api_key:
            st.error("Please enter your DeepSeek API key!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Step 1: Find artist likely to have good official channel
        with st.spinner(f"🔍 Finding {genre_input} artist with official channel..."):
            artist_result = discover_artist_with_channel(
                genre_input, 
                llm, 
                st.session_state.excluded_artists
            )
            
            if not artist_result["success"] or not artist_result["artist"]:
                st.error("Could not find suitable artist. Try a different genre.")
                return
            
            artist_name = artist_result["artist"]
            
            # Check for duplicates
            if artist_name in st.session_state.excluded_artists:
                st.warning(f"⚠️ {artist_name} was already suggested. Try a different genre.")
                return
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        
        if artist_result.get("channel_reason"):
            with st.expander("Why this artist has a good channel", expanded=False):
                st.write(artist_result["channel_reason"])
        
        # Step 2: FIND AND VERIFY OFFICIAL CHANNEL (CRITICAL PHASE)
        st.markdown("---")
        st.markdown("### 🔍 **Phase 1: Finding Official YouTube Channel**")
        
        with st.spinner("Searching for official YouTube channel..."):
            channel_info = verify_and_lock_channel(
                artist_name,
                llm,
                list(st.session_state.locked_channels.values())
            )
        
        if channel_info["found"]:
            st.markdown(f'<div class="channel-locked">', unsafe_allow_html=True)
            st.markdown(f"**✅ CHANNEL LOCKED:** {channel_info['name']}")
            st.markdown(f"**Confidence:** {channel_info['confidence']*100:.0f}%")
            
            if channel_info.get("url"):
                st.markdown(f"**Approx URL:** `{channel_info['url']}`")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Store locked channel
            st.session_state.locked_channels[artist_name] = channel_info
            
        else:
            st.markdown(f'<div class="channel-failed">', unsafe_allow_html=True)
            st.error("❌ Could not find official channel!")
            st.markdown("""
            **Possible reasons:**
            1. Artist doesn't have an official YouTube channel
            2. Channel name doesn't match artist name
            3. Channel is very new or not verified
            
            **Suggestions:**
            - Try a more mainstream artist
            - Check if artist has VEVO channel
            - Try searching manually on YouTube
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            return
        
        # Step 3: Get songs for this artist
        with st.spinner("Getting popular songs..."):
            songs = get_songs_for_artist(artist_name, genre_input, llm)
        
        if not songs:
            st.warning("Could not get song list. Using generic titles.")
            songs = [
                {"title": "Popular Hit"},
                {"title": "Latest Release"},
                {"title": "Fan Favorite"}
            ]
        
        # Step 4: CHANNEL-LOCKED SONG SEARCH
        st.markdown("---")
        st.markdown(f"### 🎵 **Phase 2: Channel-Locked Song Search**")
        st.info(f"**All searches locked to:** {channel_info['name']}")
        
        videos_found = 0
        video_results = []
        
        # Create columns for song display
        cols = st.columns(3)
        
        for idx, song in enumerate(songs):
            with cols[idx]:
                st.markdown(f"**Song {idx+1}**")
                st.code(song["title"])
                
                # CHANNEL-LOCKED SEARCH
                with st.spinner(f"Searching in {channel_info['name']}..."):
                    video_result = search_within_channel_only(
                        song["title"],
                        artist_name,
                        channel_info["name"]
                    )
                
                video_results.append(video_result)
                
                if video_result.get("found"):
                    videos_found += 1
                    
                    # Display video
                    try:
                        st.video(video_result["url"])
                    except:
                        # Show thumbnail as fallback
                        video_id = video_result["url"].split("v=")[1].split("&")[0]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                    
                    # Video details
                    with st.expander("Video Details", expanded=False):
                        st.write(f"**Title:** {video_result['title'][:50]}...")
                        st.write(f"**Channel:** {video_result['channel']}")
                        st.write(f"**Match Score:** {video_result.get('score', 0)}/100")
                        if video_result.get("note"):
                            st.success(video_result["note"])
                    
                    # Watch button
                    st.markdown(
                        f'<a href="{video_result["url"]}" target="_blank">'
                        '<button style="background-color: #FF0000; color: white; '
                        'border: none; padding: 8px 16px; border-radius: 4px; '
                        'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
                
                else:
                    st.warning("Video not found in locked channel")
                    if video_result.get("suggestion"):
                        st.caption(video_result["suggestion"])
        
        # Store session
        session_data = {
            "genre": genre_input,
            "artist": artist_name,
            "channel": channel_info,
            "songs": songs,
            "video_results": video_results,
            "videos_found": videos_found,
            "timestamp": time.time()
        }
        
        st.session_state.discovery_sessions.append(session_data)
        st.session_state.excluded_artists.append(artist_name)
        
        # Summary
        st.markdown("---")
        
        if videos_found == 3:
            st.success(f"✅ Perfect! Found all 3 songs in {channel_info['name']}")
        elif videos_found > 0:
            st.warning(f"⚠️ Found {videos_found}/3 songs in the locked channel")
            st.info(f"**Note:** Some songs might not have official videos, or titles don't match exactly.")
        else:
            st.error(f"❌ No songs found in the locked channel")
            st.markdown(f"""
            **Possible issues:**
            1. Channel {channel_info['name']} might not have official music videos
            2. Song titles might not match YouTube video titles
            3. Channel might be the wrong one
            
            **Next steps:**
            1. Check channel manually: Search "{artist_name} official" on YouTube
            2. Verify the channel has the music note badge (♪)
            3. Try a different artist
            """)
    
    # Session history
    if st.session_state.discovery_sessions:
        st.markdown("---")
        st.markdown("### 📚 Recent Sessions")
        
        for session in reversed(st.session_state.discovery_sessions[-3:]):
            with st.expander(
                f"{session['genre'][:15]} → {session['artist'][:15]} "
                f"({session['videos_found']}/3 videos)", 
                expanded=False
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Channel:** {session['channel'].get('name', 'Unknown')}")
                    st.write(f"**Confidence:** {session['channel'].get('confidence', 0)*100:.0f}%")
                with col2:
                    st.write(f"**Songs found:** {session['videos_found']}/3")
                
                if session['video_results']:
                    st.write("**Videos:**")
                    for i, video in enumerate(session['video_results']):
                        if video.get("found"):
                            st.write(f"{i+1}. ✅ {video.get('title', 'Unknown')[:40]}...")
                        else:
                            st.write(f"{i+1}. ❌ Not found in locked channel")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <strong>🔒 Channel-Locked Architecture</strong><br>
    1. Find official channel first → 2. Lock all searches to that channel → 3. Never show wrong-channel videos<br>
    <em>Accuracy over quantity. Official channels only.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
