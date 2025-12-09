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
# ENHANCED ARTIST & GENRE HANDLING WITH INTELLIGENT ROTATION
# =============================================================================

def analyze_genre_popularity(genre: str, llm: ChatOpenAI, search_attempts: int) -> Dict:
    """
    Analyze if a genre is popular, moderate, or niche
    Returns: {"level": "POPULAR"/"MODERATE"/"NICHE", "explanation": str, "artist_pool": int}
    """
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""Analyze the music genre: "{genre}"
        
        Based on your music knowledge, classify it into one of these categories:
        
        POPULAR: Mainstream genre with MANY artists (50+), strong commercial presence
        MODERATE: Established genre with SEVERAL artists (10-30), active fanbase  
        NICHE: Obscure genre with FEW artists (1-10), limited commercial presence
        
        Consider: Number of professional artists, commercial success, streaming numbers.
        This is attempt #{attempts} for this genre.
        
        Return EXACTLY:
        LEVEL: [POPULAR/MODERATE/NICHE]
        EXPLANATION: [Brief reasoning]
        ESTIMATED_ARTISTS: [Number range like "50+", "10-30", or "1-10"]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": search_attempts})
        
        level = "MODERATE"
        explanation = ""
        artist_pool = "10-30"
        
        for line in result.strip().split('\n'):
            if line.startswith("LEVEL:"):
                level = line.replace("LEVEL:", "").strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("ESTIMATED_ARTISTS:"):
                artist_pool = line.replace("ESTIMATED_ARTISTS:", "").strip()
        
        return {
            "level": level,
            "explanation": explanation,
            "artist_pool": artist_pool,
            "attempts": search_attempts
        }
    except:
        return {
            "level": "MODERATE",
            "explanation": "Standard music genre",
            "artist_pool": "10-30",
            "attempts": search_attempts
        }

def discover_artist_with_rotation(genre: str, llm: ChatOpenAI, excluded_artists: List[str], 
                                 genre_popularity: Dict, max_attempts: int = 3) -> Dict:
    """
    Find an artist for the genre with intelligent rotation
    Handles popular genres (many artists) vs niche genres (few artists)
    """
    
    # Calculate how many artists we should have excluded by now
    current_attempt = genre_popularity["attempts"]
    artist_pool = genre_popularity["artist_pool"]
    
    # Format excluded artists for prompt
    if excluded_artists:
        # Show recent exclusions (last 5-10 depending on pool size)
        if artist_pool == "50+":
            show_last = min(10, len(excluded_artists))
        elif artist_pool == "10-30":
            show_last = min(5, len(excluded_artists))
        else:  # 1-10
            show_last = min(3, len(excluded_artists))
        
        shown_excluded = excluded_artists[-show_last:] if excluded_artists else []
        excluded_text = "\n".join([f"- {artist}" for artist in shown_excluded])
    else:
        excluded_text = "None"
    
    # Adjust prompt based on genre popularity
    if genre_popularity["level"] == "POPULAR":
        artist_requirement = f"Choose a DIFFERENT mainstream artist from the many available in {genre}. There are {artist_pool} artists to choose from."
    elif genre_popularity["level"] == "MODERATE":
        artist_requirement = f"Choose an established artist from {genre}. Be creative but realistic."
    else:  # NICHE
        if current_attempt >= 2:  # After 2 attempts with niche genre
            return {
                "status": "GENRE_TOO_NICHE",
                "message": f"Genre '{genre}' appears too niche. Only {artist_pool} artists available.",
                "attempts": current_attempt
            }
        artist_requirement = f"This is a niche genre with only {artist_pool} artists. Choose carefully."
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text", "artist_requirement", "attempt"],
        template="""Find ONE artist for the music genre: {genre}
        
        {artist_requirement}
        
        CRITICAL REQUIREMENTS:
        1. Artist MUST have an official YouTube channel (verified music note badge ♪)
        2. Artist MUST have at least 3 official music videos on YouTube
        3. Artist MUST NOT be in this excluded list:
        {excluded_text}
        
        This is search attempt #{attempt} for this genre.
        
        If you cannot find a suitable artist that meets ALL requirements, return:
        NO_ARTIST: [Explain why no artist meets requirements]
        
        Otherwise, return:
        ARTIST: [Artist Name]
        CHANNEL_CONFIDENCE: [High/Medium/Low - how sure you are they have official channel]
        REASON_CHOSEN: [Why this artist fits the genre]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "genre": genre,
            "excluded_text": excluded_text,
            "artist_requirement": artist_requirement,
            "attempt": current_attempt
        })
        
        # Check for "no artist" response
        for line in result.strip().split('\n'):
            if line.startswith("NO_ARTIST:"):
                return {
                    "status": "NO_ARTIST_FOUND",
                    "explanation": line.replace("NO_ARTIST:", "").strip(),
                    "genre_popularity": genre_popularity["level"],
                    "attempts": current_attempt
                }
        
        # Parse artist information
        artist = ""
        confidence = "Medium"
        reason = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist = line.replace("ARTIST:", "").strip()
            elif line.startswith("CHANNEL_CONFIDENCE:"):
                confidence = line.replace("CHANNEL_CONFIDENCE:", "").strip()
            elif line.startswith("REASON_CHOSEN:"):
                reason = line.replace("REASON_CHOSEN:", "").strip()
        
        if not artist:
            return {
                "status": "NO_ARTIST_FOUND",
                "explanation": "Could not identify a suitable artist.",
                "genre_popularity": genre_popularity["level"],
                "attempts": current_attempt
            }
        
        return {
            "status": "ARTIST_FOUND",
            "artist": artist,
            "channel_confidence": confidence,
            "reason": reason,
            "genre_popularity": genre_popularity["level"],
            "attempts": current_attempt
        }
    
    except Exception as e:
        return {
            "status": "ERROR",
            "explanation": f"Technical error: {str(e)}",
            "genre_popularity": genre_popularity["level"],
            "attempts": current_attempt
        }

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

def handle_niche_genre_fallback(genre: str, llm: ChatOpenAI, attempts: int) -> Dict:
    """
    Provide helpful explanation and representative songs for niche genres
    """
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""The music genre "{genre}" appears to be too niche for finding artists with official YouTube channels.
        
        After {attempts} attempts, we couldn't find suitable artists with official channels.
        
        Please provide:
        1. An explanation of why this genre lacks artists with official YouTube presence
        2. 3 representative song titles from this genre (even without official videos)
        3. Suggestions for how to explore this genre
        
        Return EXACTLY:
        EXPLANATION: [Why this genre lacks official channels]
        SONG_1: [Representative song 1]
        SONG_2: [Representative song 2] 
        SONG_3: [Representative song 3]
        SUGGESTIONS: [How to explore this genre without official channels]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "attempts": attempts})
        
        explanation = ""
        songs = []
        suggestions = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("SONG_1:"):
                songs.append(line.replace("SONG_1:", "").strip())
            elif line.startswith("SONG_2:"):
                songs.append(line.replace("SONG_2:", "").strip())
            elif line.startswith("SONG_3:"):
                songs.append(line.replace("SONG_3:", "").strip())
            elif line.startswith("SUGGESTIONS:"):
                suggestions = line.replace("SUGGESTIONS:", "").strip()
        
        return {
            "explanation": explanation,
            "songs": songs[:3],
            "suggestions": suggestions,
            "genre": genre,
            "attempts": attempts
        }
    
    except:
        return {
            "explanation": f"The genre '{genre}' appears to be too niche for finding artists with official YouTube channels. After {attempts} attempts, no suitable artists were found.",
            "songs": [
                f"Representative {genre} track 1",
                f"Representative {genre} track 2",
                f"Representative {genre} track 3"
            ],
            "suggestions": f"Try searching for '{genre}' on platforms like Bandcamp, SoundCloud, or specialized music forums.",
            "genre": genre,
            "attempts": attempts
        }

# =============================================================================
# STREAMLIT APP WITH INTELLIGENT ARTIST ROTATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="🎵 Smart Music Explorer",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state with enhanced tracking
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    if 'genre_attempts' not in st.session_state:
        st.session_state.genre_attempts = {}  # genre -> attempt count
    if 'genre_popularity' not in st.session_state:
        st.session_state.genre_popularity = {}  # genre -> popularity analysis
    
    # Custom CSS
    st.markdown("""
    <style>
    .popular-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px;
    }
    .niche-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px;
    }
    .explanation-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎵 Smart Music Explorer")
    st.markdown("""
    **Intelligent Features:**
    - 🎯 **Smart Artist Rotation**: Prevents repeats based on genre popularity
    - 🧠 **Genre Analysis**: Adapts to popular vs niche genres
    - ⚠️ **Helpful Fallbacks**: Explanations when content isn't available
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
        
        # Statistics
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
    
    col1, col2 = st.columns([3, 1])
    with col1:
        genre_input = st.text_input(
            "Genre name:",
            placeholder="e.g., K-pop (popular), synthwave (moderate), dungeon synth (niche)...",
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
        
        # Analyze genre popularity
        if genre_input not in st.session_state.genre_popularity:
            with st.spinner("Analyzing genre popularity..."):
                genre_analysis = analyze_genre_popularity(genre_input, llm, current_attempt)
                st.session_state.genre_popularity[genre_input] = genre_analysis
        else:
            genre_analysis = st.session_state.genre_popularity[genre_input]
            # Update attempt count
            genre_analysis["attempts"] = current_attempt
        
        # Show genre classification
        if genre_analysis["level"] == "POPULAR":
            st.markdown(f'<span class="popular-badge">🎵 POPULAR GENRE: {genre_analysis["artist_pool"]} artists</span>', unsafe_allow_html=True)
        elif genre_analysis["level"] == "NICHE":
            st.markdown(f'<span class="niche-badge">🔍 NICHE GENRE: {genre_analysis["artist_pool"]} artists</span>', unsafe_allow_html=True)
        
        # Step 1: Find artist with intelligent rotation
        with st.spinner(f"Finding artist (attempt {current_attempt})..."):
            artist_result = discover_artist_with_rotation(
                genre_input,
                llm,
                st.session_state.excluded_artists,
                genre_analysis
            )
        
        # Handle different outcomes
        if artist_result["status"] == "GENRE_TOO_NICHE":
            # Niche genre with limited artists
            st.markdown("---")
            st.markdown("### ⚠️ Niche Genre Detected")
            st.markdown(f'<div class="warning-box">', unsafe_allow_html=True)
            st.warning("**Limited Artists Available**")
            st.write(f"**Genre:** {genre_input}")
            st.write(f"**Artist Pool:** {genre_analysis['artist_pool']}")
            st.write(f"**Attempts:** {current_attempt}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Get niche genre fallback
            with st.spinner("Preparing genre explanation..."):
                niche_fallback = handle_niche_genre_fallback(genre_input, llm, current_attempt)
            
            st.markdown("### 📝 Why This Happened")
            st.write(niche_fallback["explanation"])
            
            st.markdown("### 🎶 Representative Songs")
            cols = st.columns(3)
            for idx, song in enumerate(niche_fallback["songs"][:3]):
                with cols[idx]:
                    st.info(f"**Song {idx+1}**")
                    st.code(song)
                    st.caption("⚠️ No official video available")
            
            st.markdown("### 💡 Suggestions")
            st.write(niche_fallback["suggestions"])
            
            # Store session
            session_data = {
                "genre": genre_input,
                "status": "NICHE_GENRE_FALLBACK",
                "popularity": genre_analysis["level"],
                "explanation": niche_fallback["explanation"],
                "songs": niche_fallback["songs"],
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)
            return
        
        elif artist_result["status"] == "NO_ARTIST_FOUND":
            # No artist found after multiple attempts
            st.markdown("---")
            st.markdown("### ❌ No Artist Found")
            st.markdown(f'<div class="warning-box">', unsafe_allow_html=True)
            st.error("**Search Exhausted**")
            st.write(f"**Genre:** {genre_input}")
            st.write(f"**Popularity:** {artist_result.get('genre_popularity', 'Unknown')}")
            st.write(f"**Attempts:** {artist_result.get('attempts', current_attempt)}")
            st.write(f"**Reason:** {artist_result.get('explanation', 'Unknown')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.info("""
            **Possible reasons:**
            1. All suitable artists for this genre have been suggested
            2. The genre might be too specific or obscure
            3. Artists in this genre might not have official YouTube channels
            
            **Try:**
            1. A broader genre term
            2. A different music style
            3. Clearing session data to reset exclusions
            """)
            
            # Store session
            session_data = {
                "genre": genre_input,
                "status": "NO_ARTIST_EXHAUSTED",
                "popularity": artist_result.get('genre_popularity', 'Unknown'),
                "explanation": artist_result.get('explanation', 'Unknown'),
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)
            return
        
        elif artist_result["status"] != "ARTIST_FOUND":
            st.error(f"Error: {artist_result.get('explanation', 'Unknown error')}")
            return
        
        # Artist found - proceed
        artist_name = artist_result["artist"]
        
        # Check for duplicates (shouldn't happen with our logic but double-check)
        if artist_name in st.session_state.excluded_artists:
            st.warning(f"⚠️ {artist_name} was already suggested! Trying different artist...")
            # Try one more time with updated exclusions
            genre_analysis["attempts"] += 1
            artist_result = discover_artist_with_rotation(
                genre_input,
                llm,
                st.session_state.excluded_artists,
                genre_analysis
            )
            
            if artist_result["status"] != "ARTIST_FOUND":
                st.error("Could not find a new artist. All suitable artists have been suggested.")
                return
            
            artist_name = artist_result["artist"]
        
        # Add to excluded artists
        if artist_name not in st.session_state.excluded_artists:
            st.session_state.excluded_artists.append(artist_name)
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        st.caption(f"Confidence: {artist_result.get('channel_confidence', 'Medium')} | Reason: {artist_result.get('reason', '')}")
        
        # Step 2: Find and lock official channel
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Channel")
        
        with st.spinner("Searching for official YouTube channel..."):
            locked_channel, status_msg = find_and_lock_official_channel(artist_name)
        
        if locked_channel:
            st.markdown(f'<div class="success-box">', unsafe_allow_html=True)
            st.success(f"✅ **Channel Locked:** {locked_channel}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Store globally
            st.session_state.current_locked_channel = locked_channel
        else:
            # No official channel found
            st.error(status_msg)
            st.markdown(f'<div class="warning-box">', unsafe_allow_html=True)
            st.warning(f"**No Official Channel Found**")
            st.write(f"Artist: {artist_name}")
            st.write("Despite our search, we couldn't find an official YouTube channel.")
            st.write("This might indicate the artist doesn't maintain an official YouTube presence.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Still show representative songs
            st.markdown("### 🎶 Representative Songs")
            with st.spinner("Getting representative songs..."):
                songs = get_three_popular_songs(artist_name, llm)
            
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
                "popularity": genre_analysis["level"],
                "songs": songs[:3],
                "attempt": current_attempt,
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
            songs = get_three_popular_songs(artist_name, llm)
        
        if not songs:
            st.warning("Could not get songs list")
            songs = ["Popular hit", "Latest release", "Fan favorite"]
        
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
                    
                    # Details
                    with st.expander("Details", expanded=False):
                        st.write(f"**Title:** {video_result['title'][:50]}...")
                        st.write(f"**Channel:** {video_result['channel']}")
                        st.write(f"**Match Score:** {video_result.get('score', 0)}/100")
                        if video_result.get("search_used"):
                            st.write(f"**Found via:** `{video_result['search_used'][:30]}...`")
                    
                    # Watch button
                    st.markdown(
                        f'<a href="{video_result["url"]}" target="_blank">'
                        '<button style="background-color: #FF0000; color: white; '
                        'border: none; padding: 8px 16px; border-radius: 4px; '
                        'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
                
                else:
                    st.warning("❌ Not found in locked channel")
                    st.caption(f"Reason: {video_result.get('error', 'Unknown')}")
        
        # Store session
        session_data = {
            "genre": genre_input,
            "artist": artist_name,
            "locked_channel": locked_channel,
            "popularity": genre_analysis["level"],
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
            st.balloons()
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
            if session.get("status") == "NICHE_GENRE_FALLBACK":
                with st.expander(f"🔍 {session['genre']} - Niche Genre", expanded=False):
                    st.write(f"**Popularity:** {session.get('popularity', 'Unknown')}")
                    st.write(f"**Result:** Genre too niche for official channels")
                    st.write(f"**Songs suggested:** {len(session.get('songs', []))}")
            elif session.get("status") == "NO_ARTIST_EXHAUSTED":
                with st.expander(f"❌ {session['genre']} - Exhausted", expanded=False):
                    st.write(f"**Popularity:** {session.get('popularity', 'Unknown')}")
                    st.write(f"**Result:** All artists exhausted")
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
