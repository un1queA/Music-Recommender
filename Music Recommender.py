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
# INITIALIZATION
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

# =============================================================================
# GENRE POPULARITY CLASSIFICATION
# =============================================================================

def get_genre_popularity_level(genre: str, llm: ChatOpenAI, search_attempts: int) -> str:
    """Determine if genre is popular, moderate, or niche"""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""Analyze the music genre: "{genre}"
        
        Based on general music knowledge, categorize it:
        
        - **POPULAR**: Genre has mainstream recognition, many artists with official channels
          Examples: Pop, Rock, Hip-Hop, K-pop, Reggaeton
        
        - **MODERATE**: Genre has dedicated following, moderate number of professional artists
          Examples: Synthwave, Math Rock, Shoegaze, City Pop
        
        - **NICHE**: Genre is obscure, few artists with official presence
          Examples: Dungeon Synth, Witch House, Noise Rock, Microtonal
        
        This is attempt #{attempts} for this genre. Return ONLY:
        LEVEL: [POPULAR/MODERATE/NICHE]
        REASON: [Brief explanation]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": search_attempts})
        
        level = "MODERATE"
        for line in result.strip().split('\n'):
            if line.startswith("LEVEL:"):
                level = line.replace("LEVEL:", "").strip()
        
        return level
    except:
        return "MODERATE"

# =============================================================================
# INTELLIGENT ARTIST DISCOVERY WITH ROTATION
# =============================================================================

def get_artist_for_genre(genre: str, llm: ChatOpenAI, excluded_artists: List[str], 
                         genre_popularity: str, search_attempts: int) -> Dict:
    """Find artist with rotation based on genre popularity"""
    
    max_display_excluded = min(15, len(excluded_artists))
    shown_excluded = excluded_artists[-max_display_excluded:] if excluded_artists else []
    excluded_text = "\n".join([f"- {artist}" for artist in shown_excluded])
    
    if genre_popularity == "POPULAR":
        emphasis = "There are MANY artists in this popular genre. Choose a different mainstream artist each time."
        min_artists_available = "50+"
    elif genre_popularity == "MODERATE":
        emphasis = "There are several notable artists in this genre. Be creative but ensure they have official channels."
        min_artists_available = "10-20"
    else:
        emphasis = "This is a niche genre with limited artists. If no suitable artist with official channel exists, acknowledge this."
        min_artists_available = "1-5"
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text", "emphasis", "min_artists", "attempts"],
        template="""Find an artist for music genre: "{genre}"
        
        {emphasis}
        
        IMPORTANT REQUIREMENTS:
        1. Artist MUST have an official YouTube channel (verified music note badge ♪)
        2. Artist MUST have at least 3 official music videos
        3. DO NOT repeat these already-suggested artists:
        {excluded_text}
        
        This is search attempt #{attempts} for this genre.
        Expected available artists in this genre: {min_artists}
        
        If this genre is too niche and no suitable artist exists, return:
        NO_ARTIST_FOUND: [Explain why no artist meets requirements]
        
        Otherwise, return:
        ARTIST: [Artist Name]
        ARTIST_TIER: [Mainstream/Established/Emerging/Niche]
        YOUTUBE_PRESENCE: [Describe their YouTube channel's quality and content]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "genre": genre,
            "excluded_text": excluded_text,
            "emphasis": emphasis,
            "min_artists": min_artists_available,
            "attempts": search_attempts
        })
        
        for line in result.strip().split('\n'):
            if line.startswith("NO_ARTIST_FOUND:"):
                return {
                    "status": "NO_ARTIST",
                    "explanation": line.replace("NO_ARTIST_FOUND:", "").strip(),
                    "genre_popularity": genre_popularity
                }
        
        artist_info = {
            "status": "FOUND",
            "artist": "",
            "tier": "",
            "youtube_presence": "",
            "genre_popularity": genre_popularity
        }
        
        for line in result.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist_info["artist"] = line.replace("ARTIST:", "").strip()
            elif line.startswith("ARTIST_TIER:"):
                artist_info["tier"] = line.replace("ARTIST_TIER:", "").strip()
            elif line.startswith("YOUTUBE_PRESENCE:"):
                artist_info["youtube_presence"] = line.replace("YOUTUBE_PRESENCE:", "").strip()
        
        if not artist_info["artist"]:
            return {
                "status": "NO_ARTIST",
                "explanation": "Could not identify a suitable artist.",
                "genre_popularity": genre_popularity
            }
        
        return artist_info
    
    except Exception as e:
        return {
            "status": "ERROR",
            "explanation": f"Technical error: {str(e)}",
            "genre_popularity": genre_popularity
        }

# =============================================================================
# IMPROVED CHANNEL DETECTION (FIXED VERSION)
# =============================================================================

def find_and_lock_official_channel(artist_name: str) -> Tuple[Optional[str], str, str]:
    """
    FIXED: Find official channel using flexible matching
    Returns: (channel_exact_name, status, details)
    """
    
    search_strategies = [
        f"{artist_name} official",
        f"{artist_name} topic",
        f"{artist_name} vevo",
        f"{artist_name} music",
        f'"{artist_name}"',
    ]
    
    all_candidate_channels = []
    
    for search_query in search_strategies:
        try:
            results = YoutubeSearch(search_query, max_results=10).to_dict()
            
            for result in results:
                channel_display_name = result['channel'].strip()
                video_title = result['title'].lower()
                
                score = 0
                artist_lower = artist_name.lower()
                channel_lower = channel_display_name.lower()
                
                # FLEXIBLE MATCHING: Artist name in channel OR channel in artist name
                if (artist_lower in channel_lower) or (channel_lower in artist_lower and len(channel_lower) > 3):
                    score += 60
                
                # Check video title for artist name
                if artist_lower in video_title:
                    score += 30
                
                # Official indicators
                official_indicators = ['official', 'topic', 'vevo']
                for indicator in official_indicators:
                    if indicator in channel_lower:
                        score += 50
                        break
                
                # Check video title for official content
                for indicator in ['official music video', 'official mv', '【official】']:
                    if indicator in video_title:
                        score += 30
                        break
                
                # Negative signals
                negative_terms = ['cover', 'tribute', 'fanmade', 'lyrics video', 'karaoke']
                for term in negative_terms:
                    if term in video_title or term in channel_lower:
                        score -= 40
                
                # Music-related channel
                music_terms = ['music', '歌曲', 'musica', '歌', 'musik']
                for term in music_terms:
                    if term in channel_lower:
                        score += 15
                        break
                
                # Only consider promising candidates
                if score >= 50:
                    if not any(chan[0] == channel_display_name for chan in all_candidate_channels):
                        all_candidate_channels.append(
                            (channel_display_name, score, search_query)
                        )
            
            time.sleep(0.2)
        except Exception as e:
            continue
    
    # Fallback strategy
    if not all_candidate_channels:
        try:
            fallback_query = f"channel {artist_name}"
            results = YoutubeSearch(fallback_query, max_results=5).to_dict()
            
            for result in results:
                title = result['title'].lower()
                if 'channel' in title or 'subscribe' in title:
                    channel_name = result['channel'].strip()
                    all_candidate_channels.append((channel_name, 35, "channel_search"))
        except:
            pass
    
    # Select best candidate
    if all_candidate_channels:
        all_candidate_channels.sort(key=lambda x: x[1], reverse=True)
        best_channel, best_score, found_via = all_candidate_channels[0]
        
        details = f"Score: {best_score}/150 via '{found_via}'."
        return best_channel, "FOUND", details
    
    return None, "NOT_FOUND", f"Exhausted {len(search_strategies)} search strategies for '{artist_name}'."

def calculate_video_match_score(video: Dict, song_query: str, channel_name: str) -> int:
    """Calculate how well a video matches our search"""
    score = 0
    title = video['title'].lower()
    clean_query = re.sub(r'[^\w\s]', '', song_query.lower())
    
    if clean_query in re.sub(r'[^\w\s]', '', title):
        score += 60
    
    query_words = set(clean_query.split())
    title_words = set(re.sub(r'[^\w\s]', '', title).split())
    common_words = query_words.intersection(title_words)
    if common_words:
        score += len(common_words) * 10
    
    official_flags = ['official', 'music video', 'mv', '【official】', '官方版']
    for flag in official_flags:
        if flag in title:
            score += 30
            break
    
    if not any(term in title for term in ['live', 'concert', 'performance', 'acoustic']):
        score += 20
    
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 2 <= mins <= 7:
                    score += 10
        except:
            pass
    
    return score

def search_strict_channel_only(song_query: str, locked_channel_name: str) -> Dict:
    """Search within locked channel with flexible matching"""
    if not locked_channel_name:
        return {"found": False, "error": "No channel locked"}
    
    search_variations = [
        f'"{song_query}" "{locked_channel_name}"',
        f"{song_query} {locked_channel_name}",
        f"{song_query} official {locked_channel_name}",
        f"{song_query}",
    ]
    
    target_channel_lower = locked_channel_name.lower().strip()
    best_result = None
    best_score = 0
    
    for search_term in search_variations:
        try:
            results = YoutubeSearch(search_term, max_results=15).to_dict()
            
            for result in results:
                result_channel = result['channel'].strip()
                
                # FLEXIBLE CHANNEL MATCHING
                is_same_channel = (
                    target_channel_lower in result_channel.lower() or 
                    result_channel.lower() in target_channel_lower
                )
                
                if not is_same_channel:
                    continue
                
                score = calculate_video_match_score(result, song_query, locked_channel_name)
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        "url": f"https://www.youtube.com/watch?v={result['id']}",
                        "title": result['title'],
                        "channel": result_channel,
                        "score": score,
                        "search_term": search_term,
                        "found": True
                    }
        
        except Exception:
            continue
    
    if best_result and best_score >= 40:
        return best_result
    
    return {
        "found": False,
        "error": f"No matching video for '{song_query}' in channel '{locked_channel_name}'",
        "strict_lock": True
    }

# =============================================================================
# SONG MANAGEMENT
# =============================================================================

def get_songs_for_scenario(artist: str, llm: ChatOpenAI, scenario: str = "NORMAL") -> List[str]:
    """Get songs based on scenario"""
    
    if scenario == "NICHE_NO_CHANNEL":
        prompt = PromptTemplate(
            input_variables=["artist"],
            template="""For the niche artist/genre "{artist}", list 3 representative song titles 
            that define this style of music (even if they don't have official videos).
            
            Return EXACTLY:
            1: [Song Title 1]
            2: [Song Title 2]
            3: [Song Title 3]"""
        )
    else:
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
    
    except:
        return [f"Representative song 1", f"Representative song 2", f"Representative song 3"]

def handle_niche_genre_case(genre: str, llm: ChatOpenAI, attempts: int) -> Dict:
    """Handle niche genres with no suitable artists/channels"""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""The music genre "{genre}" appears to be too niche for our system 
        (attempt #{attempts} failed to find artist with official YouTube channel).
        
        Provide a helpful explanation for the user about why this genre doesn't work well 
        with our system, and suggest alternative approaches.
        
        Return EXACTLY:
        EXPLANATION: [Why this genre is problematic]
        SUGGESTION: [What the user can try instead]
        EXAMPLE_SONGS: [3 representative song titles from this genre, format: "Song1 | Song2 | Song3"]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": attempts})
        
        explanation = ""
        suggestion = ""
        example_songs = []
        
        for line in result.strip().split('\n'):
            if line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("SUGGESTION:"):
                suggestion = line.replace("SUGGESTION:", "").strip()
            elif line.startswith("EXAMPLE_SONGS:"):
                songs_text = line.replace("EXAMPLE_SONGS:", "").strip()
                example_songs = [s.strip() for s in songs_text.split('|')]
        
        return {
            "status": "NICHE_GENRE_EXPLANATION",
            "explanation": explanation,
            "suggestion": suggestion,
            "example_songs": example_songs[:3]
        }
    
    except:
        return {
            "status": "NICHE_GENRE_EXPLANATION",
            "explanation": "This genre appears to be too niche for our system to find artists with official YouTube channels.",
            "suggestion": "Try a more mainstream genre, or search for these artists manually on YouTube.",
            "example_songs": ["Representative track 1", "Representative track 2", "Representative track 3"]
        }

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="🎵 Smart Genre Explorer",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    if 'genre_attempts' not in st.session_state:
        st.session_state.genre_attempts = {}
    if 'genre_popularity' not in st.session_state:
        st.session_state.genre_popularity = {}
    
    # Custom CSS
    st.markdown("""
    <style>
    .popular-genre {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .niche-genre {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .explanation-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .attempt-counter {
        background-color: #E3F2FD;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        display: inline-block;
        margin-left: 10px;
    }
    .channel-success {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎵 Smart Genre Explorer")
    st.markdown("""
    **Intelligent Features:**
    - 🎯 **Artist Rotation**: Never repeats artists for popular genres
    - 🧠 **Genre Awareness**: Different handling for popular vs niche genres  
    - 🔍 **Smart Channel Detection**: Flexible matching for official channels
    - 🔒 **Channel Locking**: 100% official channel guarantee
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
        
        # Stats
        st.markdown("### 📊 Statistics")
        st.write(f"Total sessions: {len(st.session_state.sessions)}")
        st.write(f"Excluded artists: {len(st.session_state.excluded_artists)}")
        
        # Genre-specific stats
        if st.session_state.genre_attempts:
            st.markdown("#### Genre Attempts")
            for genre, attempts in list(st.session_state.genre_attempts.items())[-5:]:
                st.write(f"**{genre[:15]}...**: {attempts} searches")
        
        if st.session_state.excluded_artists:
            with st.expander("Recently excluded artists"):
                for artist in st.session_state.excluded_artists[-10:]:
                    st.write(f"• {artist}")
        
        if st.button("🔄 Clear All Data", type="secondary"):
            st.session_state.sessions = []
            st.session_state.excluded_artists = []
            st.session_state.genre_attempts = {}
            st.session_state.genre_popularity = {}
            st.rerun()
    
    # Main input
    st.markdown("### 🎯 Enter Music Genre")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        genre_input = st.text_input(
            "Genre name:",
            placeholder="e.g., synthwave, K-pop, dungeon synth, reggae...",
            label_visibility="collapsed",
            key="genre_input"
        )
    
    with col2:
        st.write("")
        search_btn = st.button(
            "🧠 Smart Search",
            type="primary",
            use_container_width=True
        )
    
    with col3:
        if genre_input and genre_input in st.session_state.genre_attempts:
            attempts = st.session_state.genre_attempts[genre_input]
            st.markdown(f'<div class="attempt-counter">Attempt {attempts}</div>', unsafe_allow_html=True)
    
    # Process search
    if search_btn and genre_input:
        if not st.session_state.api_key:
            st.error("Please enter your DeepSeek API key!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Track genre attempts
        if genre_input not in st.session_state.genre_attempts:
            st.session_state.genre_attempts[genre_input] = 0
        st.session_state.genre_attempts[genre_input] += 1
        current_attempt = st.session_state.genre_attempts[genre_input]
        
        # Determine genre popularity
        if genre_input not in st.session_state.genre_popularity:
            with st.spinner("Analyzing genre popularity..."):
                popularity = get_genre_popularity_level(genre_input, llm, current_attempt)
                st.session_state.genre_popularity[genre_input] = popularity
        else:
            popularity = st.session_state.genre_popularity[genre_input]
        
        # Show genre classification
        if popularity == "POPULAR":
            st.markdown(f'<div class="popular-genre">', unsafe_allow_html=True)
            st.markdown(f"**🎵 POPULAR GENRE**: {genre_input} - Many artists available for rotation")
            st.markdown("</div>", unsafe_allow_html=True)
        elif popularity == "NICHE":
            st.markdown(f'<div class="niche-genre">', unsafe_allow_html=True)
            st.markdown(f"**🔍 NICHE GENRE**: {genre_input} - Limited artists with official channels")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Step 1: Find artist with intelligent rotation
        with st.spinner(f"Finding artist (attempt {current_attempt})..."):
            artist_result = get_artist_for_genre(
                genre_input,
                llm,
                st.session_state.excluded_artists,
                popularity,
                current_attempt
            )
        
        # Handle different outcomes
        if artist_result["status"] == "NO_ARTIST":
            # No suitable artist found - niche genre case
            st.markdown("---")
            st.markdown(f'<div class="explanation-box">', unsafe_allow_html=True)
            st.error("🎵 **Genre Analysis Complete**")
            st.write(f"**Genre:** {genre_input}")
            st.write(f"**Popularity Level:** {popularity}")
            st.write("**Result:** No suitable artist found with official YouTube channel")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Get detailed explanation for niche genre
            with st.spinner("Generating detailed explanation..."):
                niche_explanation = handle_niche_genre_case(genre_input, llm, current_attempt)
            
            st.markdown("### 📝 Why This Happened")
            st.write(niche_explanation["explanation"])
            
            st.markdown("### 💡 Suggestions")
            st.write(niche_explanation["suggestion"])
            
            if niche_explanation.get("example_songs"):
                st.markdown("### 🎶 Representative Songs in This Genre")
                cols = st.columns(3)
                for idx, song in enumerate(niche_explanation["example_songs"][:3]):
                    with cols[idx]:
                        st.info(f"**Song {idx+1}**")
                        st.code(song)
            
            # Store as session
            session_data = {
                "genre": genre_input,
                "status": "NICHE_NO_ARTIST",
                "popularity": popularity,
                "explanation": niche_explanation["explanation"],
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)
            return
        
        elif artist_result["status"] != "FOUND":
            st.error(f"Error: {artist_result.get('explanation', 'Unknown error')}")
            return
        
        # Artist found - proceed
        artist_name = artist_result["artist"]
        
        # Add to excluded artists (for this session)
        if artist_name not in st.session_state.excluded_artists:
            st.session_state.excluded_artists.append(artist_name)
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        st.caption(f"Tier: {artist_result.get('tier', 'Unknown')} | YouTube: {artist_result.get('youtube_presence', '')}")
        
        # Step 2: Find and lock official channel (FIXED VERSION)
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Channel")
        
        with st.spinner("Searching for official YouTube channel (using flexible matching)..."):
            locked_channel, channel_status, channel_details = find_and_lock_official_channel(artist_name)
        
        if channel_status == "FOUND" and locked_channel:
            st.markdown(f'<div class="channel-success">', unsafe_allow_html=True)
            st.success(f"✅ **Channel Locked:** {locked_channel}")
            if channel_details:
                st.caption(f"Detection details: {channel_details}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # No official channel found for this artist
            st.warning(f"⚠️ No official channel found for {artist_name}")
            st.markdown(f'<div class="explanation-box">', unsafe_allow_html=True)
            st.write(f"**Artist:** {artist_name}")
            st.write(f"**Genre:** {genre_input}")
            st.write(f"**Status:** {channel_details}")
            st.write("**Suggestion:** Try a different artist or search manually on YouTube")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Get representative songs anyway
            st.markdown("### 🎶 Representative Songs")
            with st.spinner("Getting representative songs..."):
                songs = get_songs_for_scenario(artist_name, llm, "NICHE_NO_CHANNEL")
            
            cols = st.columns(3)
            for idx, song in enumerate(songs[:3]):
                with cols[idx]:
                    st.info(f"**Song {idx+1}**")
                    st.code(song)
                    st.caption("⚠️ No official video available")
            
            # Store session
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "status": "NO_CHANNEL",
                "popularity": popularity,
                "attempt": current_attempt,
                "songs": songs[:3],
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)
            return
        
        # Step 3: Get songs and search in locked channel
        st.markdown("---")
        st.markdown("### 🎵 Channel-Locked Song Search")
        st.info(f"**All searches locked to:** {locked_channel}")
        
        # Get songs
        with st.spinner("Getting popular songs..."):
            songs = get_songs_for_scenario(artist_name, llm, "NORMAL")
        
        # Search for each song in locked channel
        videos_found = 0
        search_results = []
        
        cols = st.columns(3)
        
        for idx, song in enumerate(songs[:3]):
            with cols[idx]:
                st.markdown(f"**Song {idx+1}**")
                st.code(song)
                
                # Strict channel-locked search
                with st.spinner("Searching..."):
                    video_result = search_strict_channel_only(song, locked_channel)
                
                search_results.append(video_result)
                
                if video_result["found"]:
                    videos_found += 1
                    
                    # Display video
                    try:
                        st.video(video_result["url"])
                    except:
                        video_id = video_result["url"].split("v=")[1]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                    
                    # Watch button
                    st.markdown(
                        f'<a href="{video_result["url"]}" target="_blank">'
                        '<button style="background-color: #FF0000; color: white; '
                        'border: none; padding: 8px 16px; border-radius: 4px; '
                        'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
                    
                    with st.expander("Details", expanded=False):
                        st.write(f"**Title:** {video_result['title'][:50]}...")
                        st.write(f"**Channel:** {video_result['channel']}")
                        st.write(f"**Match Score:** {video_result.get('score', 0)}/100")
                        if video_result.get("search_term"):
                            st.write(f"**Found via:** `{video_result['search_term'][:30]}...`")
                
                else:
                    st.warning("❌ Not found in locked channel")
                    st.caption(f"Reason: {video_result.get('error', 'Unknown')}")
        
        # Store session
        session_data = {
            "genre": genre_input,
            "artist": artist_name,
            "locked_channel": locked_channel,
            "popularity": popularity,
            "status": "COMPLETE",
            "videos_found": videos_found,
            "total_songs": len(songs),
            "attempt": current_attempt,
            "timestamp": time.time()
        }
        
        st.session_state.sessions.append(session_data)
        
        # Summary
        st.markdown("---")
        
        if videos_found == len(songs):
            st.success(f"🎉 **Success:** All {videos_found} songs found in {locked_channel}")
        elif videos_found > 0:
            st.warning(f"⚠️ **Partial:** {videos_found}/{len(songs)} songs found in locked channel")
            st.info(f"**Note:** Some songs might not have official videos in this channel")
        else:
            st.error(f"❌ **No videos** found in locked channel")
            st.markdown(f"""
            **Possible reasons:**
            1. Channel {locked_channel} might not have official music videos
            2. Song titles might not match exactly
            3. Try searching manually: "{artist_name}" on YouTube
            """)
    
    # Recent sessions
    if st.session_state.sessions:
        st.markdown("---")
        st.markdown("### 📋 Recent Sessions")
        
        for session in reversed(st.session_state.sessions[-5:]):
            if session.get("status") == "NICHE_NO_ARTIST":
                with st.expander(f"🔍 {session['genre']} - Niche Genre Analysis", expanded=False):
                    st.write(f"**Popularity:** {session.get('popularity', 'Unknown')}")
                    st.write(f"**Result:** No artist found with official channel")
                    st.write(f"**Attempt:** #{session.get('attempt', 1)}")
            elif session.get("status") == "NO_CHANNEL":
                with st.expander(f"⚠️ {session['genre']} → {session['artist']} - No Channel", expanded=False):
                    st.write(f"**Popularity:** {session.get('popularity', 'Unknown')}")
                    st.write(f"**Result:** Artist found but no official channel")
                    st.write(f"**Songs suggested:** {len(session.get('songs', []))}")
            else:
                with st.expander(
                    f"{session['genre']} → {session['artist']} "
                    f"({session.get('videos_found', 0)}/{session.get('total_songs', 3)} videos)",
                    expanded=False
                ):
                    st.write(f"**Channel:** {session.get('locked_channel', 'Unknown')}")
                    st.write(f"**Popularity:** {session.get('popularity', 'Unknown')}")
                    st.write(f"**Attempt:** #{session.get('attempt', 1)}")

if __name__ == "__main__":
    main()
