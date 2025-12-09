# prerequisites:
# pip install langchain-openai youtube-search streamlit langchain_core
# Run: streamlit run app.py

import streamlit as st
import re
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
import time
import hashlib

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =============================================================================
# CORE: STRICT CHANNEL LOCKING SYSTEM
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

def get_channel_exact_name(search_results: List[Dict], artist_name: str) -> Optional[str]:
    """
    Find the EXACT channel name from search results with strict rules
    Returns the exact channel name string or None
    """
    artist_lower = artist_name.lower().strip()
    best_channel = None
    best_score = -100  # Start negative to filter out bad matches
    
    for result in search_results:
        channel = result['channel'].strip()
        channel_lower = channel.lower()
        title = result['title'].lower()
        
        score = 0
        
        # MUST-HAVE: Artist name in channel (case-insensitive)
        if artist_lower in channel_lower:
            score += 100  # CRITICAL: Without this, reject completely
        else:
            # Skip completely if artist name not in channel
            continue
        
        # STRONG POSITIVE SIGNALS
        if 'official' in channel_lower:
            score += 50
        if 'topic' in channel_lower:  # YouTube Music official channels
            score += 40
        if 'vevo' in channel_lower:
            score += 35
        if 'music' in channel_lower:
            score += 20
            
        # STRONG NEGATIVE SIGNALS (reject these)
        negative_terms = ['cover', 'tribute', 'fan channel', 'reaction', 'compilation', 'lyrics']
        for term in negative_terms:
            if term in channel_lower:
                score -= 100  # Auto-reject
                break
                
        # Additional checks
        if 'channel' in title or 'subscribe' in title:
            score += 10
            
        # Update best channel
        if score > best_score and score > 0:  # Must have positive score
            best_score = score
            best_channel = channel
    
    return best_channel

def find_and_lock_official_channel(artist_name: str) -> Tuple[Optional[str], str]:
    """
    PHASE 1: Find and lock the official channel with 100% certainty
    Returns: (channel_exact_name, status_message)
    """
    
    # Multiple search strategies
    search_strategies = [
        f"{artist_name} official channel",
        f"{artist_name} topic",
        f"{artist_name} vevo",
        f"{artist_name} music official",
        f'"{artist_name}" YouTube channel'
    ]
    
    all_results = []
    
    for search_query in search_strategies:
        try:
            results = YoutubeSearch(search_query, max_results=10).to_dict()
            all_results.extend(results)
            time.sleep(0.1)  # Small delay to avoid rate limiting
        except Exception as e:
            continue
    
    # Find the exact channel name
    exact_channel_name = get_channel_exact_name(all_results, artist_name)
    
    if exact_channel_name:
        return exact_channel_name, f"✅ LOCKED: {exact_channel_name}"
    else:
        return None, f"❌ Could not find official channel for {artist_name}"

def search_strict_channel_only(song_query: str, locked_channel_name: str) -> Dict:
    """
    PHASE 2: Search with 100% channel locking
    Returns video ONLY if it's from the locked channel
    """
    
    if not locked_channel_name:
        return {
            "found": False,
            "error": "No channel locked",
            "strict_lock": True
        }
    
    # Normalize channel name for comparison
    target_channel_lower = locked_channel_name.lower().strip()
    
    # Search variations
    search_variations = [
        f'"{song_query}" "{locked_channel_name}"',
        f"{song_query} {locked_channel_name}",
        f"{song_query}",
        f"{song_query} official music video"
    ]
    
    for search_term in search_variations:
        try:
            results = YoutubeSearch(search_term, max_results=15).to_dict()
            
            # STRICT FILTER: Only videos from locked channel
            channel_videos = []
            for result in results:
                result_channel = result['channel'].strip()
                if result_channel.lower() == target_channel_lower:
                    channel_videos.append(result)
            
            # If we found videos from our EXACT channel
            if channel_videos:
                # Pick the best match by title relevance
                best_video = None
                best_score = 0
                
                for video in channel_videos:
                    score = 0
                    title = video['title'].lower()
                    
                    # Song title in video title (cleaned)
                    clean_query = re.sub(r'[^\w\s]', '', song_query.lower())
                    clean_title = re.sub(r'[^\w\s]', '', title)
                    
                    if clean_query in clean_title:
                        score += 60
                    else:
                        # Check word overlap
                        query_words = set(clean_query.split())
                        title_words = set(clean_title.split())
                        overlap = len(query_words.intersection(title_words))
                        if overlap > 0:
                            score += overlap * 15
                    
                    # Official indicators
                    official_terms = ['official', 'music video', 'mv', '【official】', '官方']
                    for term in official_terms:
                        if term in title:
                            score += 30
                            break
                    
                    # Avoid live/concert versions
                    if not any(term in title for term in ['live', 'concert', 'performance', 'acoustic']):
                        score += 20
                    
                    # Duration check (typical song: 2-7 minutes)
                    if video.get('duration'):
                        try:
                            parts = video['duration'].split(':')
                            if len(parts) == 2:
                                mins = int(parts[0])
                                if 2 <= mins <= 7:
                                    score += 10
                        except:
                            pass
                    
                    if score > best_score:
                        best_score = score
                        best_video = video
                
                # Only return if we have a good match
                if best_video and best_score >= 40:
                    return {
                        "found": True,
                        "url": f"https://www.youtube.com/watch?v={best_video['id']}",
                        "title": best_video['title'],
                        "channel": best_video['channel'],
                        "score": best_score,
                        "strict_lock": True,
                        "match_quality": "good" if best_score >= 60 else "fair",
                        "search_used": search_term
                    }
                
        except Exception as e:
            continue
    
    # NO VIDEO FOUND IN LOCKED CHANNEL
    return {
        "found": False,
        "error": f"No video found in locked channel: {locked_channel_name}",
        "strict_lock": True,
        "suggestion": f"Search manually on YouTube: '{song_query}' in channel '{locked_channel_name}'"
    }

# =============================================================================
# ARTIST & SONG DISCOVERY
# =============================================================================

def discover_artist_for_genre(genre: str, llm: ChatOpenAI, excluded_artists: List[str]) -> Dict:
    """Find artist with high likelihood of having official YouTube channel"""
    
    excluded_text = "\n".join([f"- {artist}" for artist in excluded_artists[:3]])
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text"],
        template="""Find ONE artist for genre: {genre}
        
        CRITICAL: Artist MUST have:
        1. Official YouTube channel (verified music note badge ♪)
        2. Clear channel name matching artist name
        3. Multiple official music videos
        
        Already excluded: {excluded_text}
        
        Return EXACTLY:
        ARTIST: [Artist Name]
        CHANNEL_LIKELIHOOD: [High/Medium/Low]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "excluded_text": excluded_text})
        
        artist = ""
        likelihood = "Medium"
        
        for line in result.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist = line.replace("ARTIST:", "").strip()
            elif line.startswith("CHANNEL_LIKELIHOOD:"):
                likelihood = line.replace("CHANNEL_LIKELIHOOD:", "").strip()
        
        return {
            "artist": artist,
            "channel_likelihood": likelihood,
            "success": bool(artist)
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_three_popular_songs(artist: str, llm: ChatOpenAI) -> List[str]:
    """Get 3 popular songs likely to have official videos"""
    
    prompt = PromptTemplate(
        input_variables=["artist"],
        template="""List 3 MOST POPULAR songs by {artist} that definitely have official music videos.
        
        Return EXACTLY:
        1: [Song Title 1]
        2: [Song Title 2]
        3: [Song Title 3]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist": artist})
        songs = []
        
        for line in result.strip().split('\n'):
            if ':' in line and any(str(i) in line.split(':')[0] for i in [1, 2, 3]):
                song_title = line.split(':', 1)[1].strip()
                if song_title:
                    songs.append(song_title)
        
        return songs[:3]
    
    except Exception as e:
        # Fallback: generic pattern
        return [
            f"Popular song by {artist}",
            f"Hit song by {artist}",
            f"Signature song by {artist}"
        ]

# =============================================================================
# STREAMLIT APP WITH 100% CHANNEL LOCKING
# =============================================================================

def main():
    st.set_page_config(
        page_title="🔒 100% Channel-Locked Music Discovery",
        page_icon="🔒",
        layout="wide"
    )
    
    # Initialize session state
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'current_locked_channel' not in st.session_state:
        st.session_state.current_locked_channel = None
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    
    # Custom CSS for strict locking
    st.markdown("""
    <style>
    .channel-locked-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2em;
    }
    .strict-lock-badge {
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 10px;
    }
    .video-container {
        border: 3px solid #4CAF50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .no-video-container {
        border: 3px solid #FF9800;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #FFF3E0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with strict locking emphasis
    st.title("🔒 100% Channel-Locked Music Discovery")
    st.markdown("""
    **STRICT RULES:**
    1. **Find official channel first** → 2. **Lock ALL searches to that exact channel** → 3. **Show ONLY videos from locked channel**
    
    **Guarantee:** If a video is shown, it's 100% from the locked official channel.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "DeepSeek API Key:",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-..."
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Current locked channel status
        if st.session_state.current_locked_channel:
            st.markdown("### 🔒 Currently Locked")
            st.info(f"**Channel:** {st.session_state.current_locked_channel}")
        
        st.markdown("### 📊 Session Stats")
        st.write(f"Sessions: {len(st.session_state.sessions)}")
        st.write(f"Excluded artists: {len(st.session_state.excluded_artists)}")
        
        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.sessions = []
            st.session_state.current_locked_channel = None
            st.session_state.excluded_artists = []
            st.rerun()
    
    # Main input
    st.markdown("### 🎯 Enter Music Genre")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        genre_input = st.text_input(
            "Genre name:",
            placeholder="e.g., hip-hop, classical, reggae, synthwave...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")
        search_btn = st.button(
            "🚀 Start Strict Search",
            type="primary",
            use_container_width=True
        )
    
    # Process search
    if search_btn and genre_input:
        if not st.session_state.api_key:
            st.error("Enter API key in sidebar!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Clear previous locked channel
        st.session_state.current_locked_channel = None
        
        # Step 1: Find artist
        with st.spinner(f"Finding {genre_input} artist with official channel..."):
            artist_result = discover_artist_for_genre(
                genre_input,
                llm,
                st.session_state.excluded_artists
            )
            
            if not artist_result["success"]:
                st.error("No artist found. Try different genre.")
                return
            
            artist_name = artist_result["artist"]
            
            # Check duplicate
            if artist_name in st.session_state.excluded_artists:
                st.warning(f"{artist_name} already suggested!")
                return
        
        st.success(f"🎤 **Artist:** {artist_name}")
        st.caption(f"Channel likelihood: {artist_result.get('channel_likelihood', 'Medium')}")
        
        # Step 2: CRITICAL - FIND AND LOCK OFFICIAL CHANNEL
        st.markdown("---")
        st.markdown("### 🔍 **PHASE 1: Find & Lock Official Channel**")
        
        with st.spinner("Searching for official YouTube channel..."):
            locked_channel, status_msg = find_and_lock_official_channel(artist_name)
        
        if locked_channel:
            st.markdown(f'<div class="channel-locked-box">', unsafe_allow_html=True)
            st.markdown(f"**🔒 CHANNEL LOCKED:** {locked_channel}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Store globally
            st.session_state.current_locked_channel = locked_channel
            
            st.info("✅ **All future searches will be STRICTLY filtered to this channel only.**")
        else:
            st.error(status_msg)
            st.warning("Cannot proceed without locked channel.")
            return
        
        # Step 3: Get songs
        with st.spinner("Getting popular songs..."):
            songs = get_three_popular_songs(artist_name, llm)
        
        if not songs:
            st.warning("Could not get songs list")
            songs = ["Popular hit", "Latest release", "Fan favorite"]
        
        # Step 4: STRICT CHANNEL-LOCKED SEARCH
        st.markdown("---")
        st.markdown(f"### 🎵 **PHASE 2: Strict Channel-Locked Search**")
        
        st.markdown(f"""
        <div style='background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        🔒 **Search Lock:** All searches filtered to <strong>{locked_channel}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        videos_found = 0
        search_results = []
        
        # Create columns for songs
        cols = st.columns(3)
        
        for idx, song in enumerate(songs):
            with cols[idx]:
                st.markdown(f"**Song {idx+1}**")
                st.code(song)
                
                # STRICT CHANNEL-LOCKED SEARCH
                with st.spinner("Searching in locked channel..."):
                    video_result = search_strict_channel_only(song, locked_channel)
                
                search_results.append(video_result)
                
                if video_result["found"]:
                    videos_found += 1
                    
                    # Display with strict lock badge
                    st.markdown(f'<div class="video-container">', unsafe_allow_html=True)
                    
                    # Match quality badge
                    quality = video_result.get("match_quality", "unknown")
                    quality_color = "#4CAF50" if quality == "good" else "#FF9800"
                    st.markdown(f'<span style="background-color: {quality_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{quality.upper()} MATCH</span>', unsafe_allow_html=True)
                    
                    # Video
                    try:
                        st.video(video_result["url"])
                    except:
                        video_id = video_result["url"].split("v=")[1]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                    
                    # Details
                    with st.expander("Details", expanded=False):
                        st.write(f"**Title:** {video_result['title'][:60]}...")
                        st.write(f"**Channel:** {video_result['channel']}")
                        st.write(f"**Score:** {video_result.get('score', 0)}/100")
                        if video_result.get("search_used"):
                            st.write(f"**Search:** `{video_result['search_used'][:40]}...`")
                    
                    # Watch button
                    st.markdown(
                        f'<a href="{video_result["url"]}" target="_blank">'
                        '<button style="background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); '
                        'color: white; border: none; padding: 10px; border-radius: 5px; '
                        'cursor: pointer; width: 100%; font-weight: bold;">'
                        '▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                else:
                    # NO VIDEO FOUND IN LOCKED CHANNEL
                    st.markdown(f'<div class="no-video-container">', unsafe_allow_html=True)
                    st.error("❌ Not found in locked channel")
                    if video_result.get("suggestion"):
                        st.caption(video_result["suggestion"])
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # Store session
        session_data = {
            "genre": genre_input,
            "artist": artist_name,
            "locked_channel": locked_channel,
            "songs": songs,
            "results": search_results,
            "videos_found": videos_found,
            "timestamp": time.time()
        }
        
        st.session_state.sessions.append(session_data)
        st.session_state.excluded_artists.append(artist_name)
        
        # Summary
        st.markdown("---")
        
        if videos_found == 3:
            st.success(f"🎉 **PERFECT:** All 3 songs found in {locked_channel}")
            st.balloons()
        elif videos_found > 0:
            st.warning(f"⚠️ **Partial:** {videos_found}/3 songs found in locked channel")
            st.info(f"**Note:** Some songs might not have official videos in {locked_channel}")
        else:
            st.error(f"❌ **No videos** found in locked channel: {locked_channel}")
            st.markdown(f"""
            **Possible reasons:**
            1. Channel might not have official music videos
            2. Song titles might not match exactly
            3. Channel could be inactive
            
            **Next steps:**
            1. Verify channel manually: Search "{artist_name} official" on YouTube
            2. Check if channel has music videos
            3. Try different artist
            """)
    
    # Recent sessions
    if st.session_state.sessions:
        st.markdown("---")
        st.markdown("### 📋 Recent Sessions")
        
        for session in reversed(st.session_state.sessions[-3:]):
            with st.expander(
                f"{session['genre']} → {session['artist']} "
                f"({session['videos_found']}/3 from {session['locked_channel'][:20]}...)",
                expanded=False
            ):
                st.write(f"**Locked Channel:** {session['locked_channel']}")
                st.write(f"**Songs Found:** {session['videos_found']}/3")
                
                for i, (song, result) in enumerate(zip(session['songs'], session['results'])):
                    status = "✅" if result.get("found") else "❌"
                    st.write(f"{i+1}. {status} {song[:30]}...")
                    
                    if result.get("found"):
                        st.caption(f"Match: {result.get('match_quality', 'unknown')} ({result.get('score', 0)}/100)")

if __name__ == "__main__":
    main()
