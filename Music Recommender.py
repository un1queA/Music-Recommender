# prerequisites:
# pip install langchain-openai youtube-search streamlit langchain_core
# Run: streamlit run app.py

import streamlit as st
import re
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
import time

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# =============================================================================
# LLM-POWERED SONG DEDUPLICATION
# =============================================================================

def are_songs_different_llm(title1: str, title2: str, llm: ChatOpenAI) -> bool:
    """Use LLM to determine if two videos represent different songs."""
    
    prompt = PromptTemplate(
        input_variables=["title1", "title2"],
        template="""Analyze these YouTube video titles:
        Title 1: "{title1}"
        Title 2: "{title2}"
        
        Are these DIFFERENT SONGS or just different versions of the SAME SONG?
        
        Return EXACTLY:
        DIFFERENT: [YES/NO]
        REASON: [Brief explanation]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"title1": title1, "title2": title2})
        different = "NO"
        for line in result.strip().split('\n'):
            if line.startswith("DIFFERENT:"):
                different = line.replace("DIFFERENT:", "").strip().upper()
                break
        return different == "YES"
    except Exception:
        return title1 != title2

def select_best_music_videos(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """Select the best music videos using LLM to ensure different songs."""
    
    if not videos or not llm:
        return videos[:count] if videos else []
    
    if len(videos) <= count:
        return videos[:count]
    
    sorted_videos = sorted(videos, key=lambda x: x.get('score', 0), reverse=True)
    selected = []
    
    for video in sorted_videos:
        if len(selected) >= count:
            break
        
        is_different_song = True
        if selected:
            for selected_video in selected:
                if not are_songs_different_llm(video['title'], selected_video['title'], llm):
                    is_different_song = False
                    break
        
        if is_different_song:
            selected.append(video)
    
    if len(selected) < count:
        selected = sorted_videos[:count]
    
    return selected[:count]

# =============================================================================
# IMPROVED CHANNEL DETECTION FROM CODE 2
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

def get_artist_search_queries(artist_name: str, genre: str, llm: ChatOpenAI) -> List[str]:
    """Use LLM to generate optimal YouTube search queries for any artist/language"""
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""For the artist "{artist_name}" (genre: {genre}), generate 3 specific YouTube search queries
        that will best find their OFFICIAL YouTube channel.
        
        IMPORTANT: Focus on finding MUSIC channels, not news or facts channels.
        
        Consider:
        1. The artist's native language and common channel naming conventions
        2. Common official channel markers in their region/language
        3. Variations of their name that might appear in channel titles
        
        Return EXACTLY 3 queries separated by | (pipe symbol):
        QUERY_1 | QUERY_2 | QUERY_3"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist_name": artist_name, "genre": genre})
        queries = [q.strip() for q in result.split('|') if q.strip()]
        return queries[:3]
    except:
        # Fallback queries if LLM fails
        return [
            f"{artist_name} official music",
            f"{artist_name} topic",
            f"{artist_name}"
        ]

def find_and_lock_official_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Tuple[Optional[str], str, List[str]]:
    """
    IMPROVED: Find official channel using LLM-optimized search queries
    Uses structural patterns instead of language-specific text
    """
    
    # Get smart search queries from LLM
    search_queries = get_artist_search_queries(artist_name, genre, llm)
    
    all_candidates = []
    
    for search_query in search_queries:
        try:
            results = YoutubeSearch(search_query, max_results=15).to_dict()
            
            for result in results:
                channel_name = result['channel'].strip()
                channel_lower = channel_name.lower()
                title_lower = result['title'].lower()
                
                # ====== CRITICAL: FILTER OUT NON-MUSIC CHANNELS ======
                # Skip news, facts, and other non-music channels immediately
                non_music_indicators = ['news', 'facts', 'update', 'rumor', 'gossip', 'magazine']
                if any(indicator in channel_lower for indicator in non_music_indicators):
                    continue
                
                # UNIVERSAL CHANNEL SCORING (no language-specific terms)
                score = 0
                
                # 1. Channel structural patterns (universal)
                # Official channels often have clean names without extra descriptors
                name_cleanliness = 0
                clean_name = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fff\uac00-\ud7a3\u3040-\u309f\u30a0-\u30ff\u0400-\u04FF]', '', channel_name)
                if len(channel_name) == len(clean_name):
                    name_cleanliness += 20
                
                # 2. Video patterns that suggest official content
                # Official videos often have consistent formatting
                if re.search(r'[【\[\(].*[】\]\)]', result['title']):
                    score += 15  # Brackets/parentheses often indicate official markers
                
                # 3. Length patterns (official music videos are typically 2-10 minutes)
                if result.get('duration'):
                    try:
                        parts = result['duration'].split(':')
                        if len(parts) == 2:
                            mins = int(parts[0])
                            if 2 <= mins <= 10:
                                score += 25  # Typical music video length
                    except:
                        pass
                
                # 4. Artist name matching (flexible, any script)
                artist_words = set(re.findall(r'[\w\u4e00-\u9fff\uac00-\ud7a3\u3040-\u309f\u30a0-\u30ff\u0400-\u04FF]+', artist_name.lower(), re.UNICODE))
                channel_words = set(re.findall(r'[\w\u4e00-\u9fff\uac00-\ud7a3\u3040-\u309f\u30a0-\u30ff\u0400-\u04FF]+', channel_lower, re.UNICODE))
                
                common_words = artist_words.intersection(channel_words)
                if common_words:
                    score += len(common_words) * 20
                
                # 5. View count as engagement signal (universal)
                if result.get('views'):
                    views_text = result['views'].lower()
                    if 'm' in views_text or 'k' in views_text:
                        score += 15
                
                # 6. STRONG MUSIC INDICATORS
                music_indicators = ['music', 'official', 'topic', 'artist', 'vevo']
                for indicator in music_indicators:
                    if indicator in channel_lower:
                        score += 20
                        break
                
                # 7. Penalize if it looks like a compilation/fan channel
                fan_channel_indicators = ['compilation', 'mix', 'edit', 'mashup', 'best of', 'top 10']
                for indicator in fan_channel_indicators:
                    if indicator in channel_lower:
                        score -= 30
                        break
                
                # Only consider promising candidates
                if score >= 40:
                    all_candidates.append({
                        'channel': channel_name,
                        'score': score + name_cleanliness,
                        'query_used': search_query,
                        'video_title': result['title']
                    })
            
            time.sleep(0.2)
        except Exception as e:
            continue
    
    # Select best candidate
    if all_candidates:
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        best = all_candidates[0]
        
        # Verify this channel has multiple videos
        try:
            # Quick check: search for more videos from this channel
            channel_check = YoutubeSearch(best['channel'], max_results=5).to_dict()
            channel_videos = [r for r in channel_check if r['channel'].strip() == best['channel']]
            
            if len(channel_videos) >= 3:
                return best['channel'], "FOUND", search_queries
        except:
            pass
        
        return best['channel'], "FOUND", search_queries
    
    return None, "NOT_FOUND", search_queries

def discover_channel_videos(locked_channel: str, artist_name: str, min_videos: int = 20) -> List[Dict]:
    """
    UNIVERSAL: Discover videos from a channel and identify music videos using structural patterns
    Returns list of videos sorted by likelihood of being official music videos
    """
    
    if not locked_channel:
        return []
    
    all_videos = []
    
    # Search for channel content using multiple approaches
    search_attempts = [
        locked_channel,
        f"{locked_channel} music",
        f"{artist_name} {locked_channel}",
    ]
    
    for search_query in search_attempts:
        try:
            results = YoutubeSearch(search_query, max_results=30).to_dict()
            
            for result in results:
                # STRICT FILTER: Only videos from our exact channel
                if result['channel'].strip() != locked_channel:
                    continue
                
                # Score video for likelihood of being music video
                score = score_video_music_likelihood(result, artist_name)
                
                if score >= 40:  # Minimum threshold
                    all_videos.append({
                        'url': f"https://www.youtube.com/watch?v={result['id']}",
                        'title': result['title'],
                        'channel': result['channel'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score,
                        'found_via': search_query
                    })
            
            if len(all_videos) >= min_videos:
                break
                
        except Exception as e:
            continue
    
    # Remove duplicates by URL
    unique_videos = {}
    for video in all_videos:
        video_id = video['url'].split('v=')[1].split('&')[0]
        if video_id not in unique_videos:
            unique_videos[video_id] = video
    
    return sorted(unique_videos.values(), key=lambda x: x['score'], reverse=True)

def score_video_music_likelihood(video: Dict, artist_name: str) -> int:
    """
    TRULY UNIVERSAL: Score video using structural patterns only
    No language-specific terms - works for Tamil, Korean, Arabic, etc.
    """
    score = 0
    title = video['title']
    
    # 1. DURATION - Most reliable universal signal
    if video.get('duration'):
        duration = video['duration']
        # Convert duration to seconds
        try:
            parts = duration.split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = int(parts[0]), int(parts[1])
                total_seconds = minutes * 60 + seconds
                
                # Music videos are typically 2-10 minutes
                if 120 <= total_seconds <= 600:  # 2-10 minutes
                    score += 60
                elif 90 <= total_seconds <= 720:  # 1.5-12 minutes (broader range)
                    score += 40
                elif 30 <= total_seconds <= 1200:  # 0.5-20 minutes (very broad)
                    score += 20
                    
            elif len(parts) == 1:  # Just seconds (likely short clip)
                if int(parts[0]) > 180:  # Over 3 minutes
                    score += 30
        except:
            pass
    
    # 2. TITLE STRUCTURAL PATTERNS (universal)
    # Look for patterns that indicate official/structured content
    title_lower = title.lower()
    
    # Common structural markers in any language
    structural_patterns = [
        r'[\[\(][^\]\)]+[\]\)]',  # Brackets/parentheses with content
        r'\d{4}',  # Years (often in official releases)
        r'[A-Z]{2,}',  # Multiple uppercase letters (acronyms, etc.)
        r'[|•\-–—]',  # Separators
    ]
    
    for pattern in structural_patterns:
        if re.search(pattern, title):
            score += 10
            break
    
    # 3. LENGTH PATTERNS - Official videos often have medium-length titles
    title_length = len(title)
    if 20 <= title_length <= 80:
        score += 15
    elif 10 <= title_length <= 120:
        score += 10
    
    # 4. VIEW POPULARITY (if available)
    if video.get('views'):
        views_text = video['views'].lower()
        if 'm' in views_text:
            score += 25
        elif 'k' in views_text:
            score += 15
        elif 'views' in views_text:
            score += 10
    
    # 5. AVOID NON-MUSIC CONTENT (using universal signals)
    # These patterns often indicate interviews, live sessions, etc.
    non_music_indicators = [
        'live at', 'concert', 'interview', 'behind the scenes',
        'making of', 'documentary', 'backstage', 'rehearsal',
        'acoustic session', 'unplugged', 'q&a', 'talk'
    ]
    
    for indicator in non_music_indicators:
        if indicator in title_lower:
            score -= 30
            break
    
    # 6. VIDEO FORMAT HINTS
    format_indicators = [
        '4k', 'hd', '1080p', 'official audio', 'visualizer',
        'lyric video', 'lyrics', 'audio only'
    ]
    
    for indicator in format_indicators:
        if indicator in title_lower:
            score += 5
    
    return max(0, score)

# =============================================================================
# ARTIST & GENRE HANDLING FROM CODE 1 (with Code 1's UI)
# =============================================================================

def analyze_genre_popularity(genre: str, llm: ChatOpenAI, search_attempts: int) -> Dict:
    """Analyze genre for popularity."""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""Analyze the music genre: "{genre}"
        
        Return EXACTLY:
        LEVEL: [POPULAR/MODERATE/NICHE]
        EXPLANATION: [Brief reasoning]
        ARTIST_POOL: [50+/10-30/1-10]"""
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
            elif line.startswith("ARTIST_POOL:"):
                artist_pool = line.replace("ARTIST_POOL:", "").strip()
        
        return {
            "level": level,
            "explanation": explanation,
            "artist_pool": artist_pool,
            "attempts": search_attempts
        }
    except Exception:
        return {
            "level": "MODERATE",
            "explanation": "Standard music genre",
            "artist_pool": "10-30",
            "attempts": search_attempts
        }

def discover_artist_with_rotation(genre: str, llm: ChatOpenAI, excluded_artists: List[str], 
                                 genre_popularity: Dict, max_attempts: int = 3) -> Dict:
    """Find artist with intelligent rotation."""
    
    current_attempt = genre_popularity["attempts"]
    artist_pool = genre_popularity["artist_pool"]
    
    # Format excluded artists
    if excluded_artists:
        if artist_pool == "50+":
            show_last = min(10, len(excluded_artists))
        elif artist_pool == "10-30":
            show_last = min(5, len(excluded_artists))
        else:
            show_last = min(3, len(excluded_artists))
        
        shown_excluded = excluded_artists[-show_last:] if excluded_artists else []
        excluded_text = "\n".join([f"- {artist}" for artist in shown_excluded])
    else:
        excluded_text = "None"
    
    # Adjust prompt
    if genre_popularity["level"] == "POPULAR":
        artist_requirement = f"Choose a DIFFERENT mainstream artist from {genre}."
    elif genre_popularity["level"] == "MODERATE":
        artist_requirement = f"Choose an established artist from {genre}."
    else:
        if current_attempt >= 2:
            return {
                "status": "GENRE_TOO_NICHE",
                "message": f"Genre '{genre}' appears too niche.",
                "attempts": current_attempt
            }
        artist_requirement = f"This is a niche genre. Choose carefully."
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text", "artist_requirement", "attempt"],
        template="""Find ONE artist for music genre: {genre}
        
        {artist_requirement}
        Not in excluded list: {excluded_text}
        
        Return EXACTLY:
        ARTIST: [Artist Name]
        CONFIDENCE: [High/Medium/Low]
        NOTE: [Brief note]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "genre": genre,
            "excluded_text": excluded_text,
            "artist_requirement": artist_requirement,
            "attempt": current_attempt
        })
        
        artist = ""
        confidence = "Medium"
        note = ""
        
        for line in result.strip().split('\n'):
            if line.startswith("ARTIST:"):
                artist = line.replace("ARTIST:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("NOTE:"):
                note = line.replace("NOTE:", "").strip()
        
        if not artist:
            return {
                "status": "NO_ARTIST_FOUND",
                "explanation": "Could not identify artist.",
                "attempts": current_attempt
            }
        
        return {
            "status": "ARTIST_FOUND",
            "artist": artist,
            "confidence": confidence,
            "note": note,
            "attempts": current_attempt
        }
    
    except Exception as e:
        return {
            "status": "ERROR",
            "explanation": f"Error: {str(e)}",
            "attempts": current_attempt
        }

def handle_niche_genre_fallback(genre: str, llm: ChatOpenAI, attempts: int) -> Dict:
    """Provide fallback for niche genres."""
    
    prompt = PromptTemplate(
        input_variables=["genre", "attempts"],
        template="""The music genre "{genre}" appears too niche.
        
        Return EXACTLY:
        EXPLANATION: [Explanation]
        SONG_1: [Song 1]
        SONG_2: [Song 2]
        SONG_3: [Song 3]
        SUGGESTIONS: [Suggestions]"""
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
    
    except Exception:
        return {
            "explanation": f"'{genre}' is too niche for official YouTube content.",
            "songs": [f"{genre} song 1", f"{genre} song 2", f"{genre} song 3"],
            "suggestions": f"Try searching '{genre}' on Bandcamp or SoundCloud.",
            "genre": genre,
            "attempts": attempts
        }

# =============================================================================
# STREAMLIT APP FROM CODE 1 (with the clean UI you like)
# =============================================================================

def main():
    st.set_page_config(
        page_title="IAMUSIC",
        page_icon="🎵",
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
    
    # Custom CSS (from Code 1)
    st.markdown("""
    <style>
    .universal-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px;
    }
            
    }
    .artist-match-high {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    .artist-match-medium {
        background: linear-gradient(135deg, #FF9800 0%, #FFC107 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    .ai-badge {
        background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        display: inline-block;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header (from Code 1)
    st.title("🔊 I AM MUSIC")
    
    # Sidebar (from Code 1)
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
        
        if st.session_state.genre_attempts:
            st.markdown("#### Genre Attempts")
            for genre, attempts in list(st.session_state.genre_attempts.items())[-5:]:
                st.write(f"**{genre[:15]}...**: {attempts} searches")
        
        if st.button("🔄 Clear All Data", type="secondary"):
            st.session_state.sessions = []
            st.session_state.excluded_artists = []
            st.session_state.genre_attempts = {}
            st.session_state.genre_popularity = {}
            st.rerun()
    
    # Main input (from Code 1)
    st.markdown("### Enter Any Music Genre")
    
    col1, col2, col3 = st.columns([2,1,1])
    
    with col1:
        with st.form(key="search_form"):
            genre_input = st.text_input(
                "Genre name:",
                placeholder="e.g., Tamil Pop, K-pop, Reggaeton, Synthwave...",
                label_visibility="collapsed",
                key="genre_input"
            )
            
            submitted = st.form_submit_button("Search", use_container_width=True, type="primary")
    
    # Process search
    if submitted and genre_input:
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
            with st.spinner("Analyzing genre..."):
                genre_analysis = analyze_genre_popularity(genre_input, llm, current_attempt)
                st.session_state.genre_popularity[genre_input] = genre_analysis
        else:
            genre_analysis = st.session_state.genre_popularity[genre_input]
            genre_analysis["attempts"] = current_attempt
        
        # Step 1: Find artist
        with st.spinner(f"Finding {genre_input} artist..."):
            artist_result = discover_artist_with_rotation(
                genre_input,
                llm,
                st.session_state.excluded_artists,
                genre_analysis
            )
        
        # Handle niche genre case
        if artist_result["status"] == "GENRE_TOO_NICHE":
            st.markdown("---")
            st.error("🎵 **Niche Genre Detected**")
            
            with st.spinner("Preparing genre explanation..."):
                niche_fallback = handle_niche_genre_fallback(genre_input, llm, current_attempt)
            
            st.write(niche_fallback["explanation"])
            
            st.markdown("### 🎶 Representative Songs")
            cols = st.columns(3)
            for idx, song in enumerate(niche_fallback["songs"][:3]):
                with cols[idx]:
                    st.info(f"**Song {idx+1}**")
                    st.code(song)
                    st.caption("⚠️ No official video available")
            
            session_data = {
                "genre": genre_input,
                "status": "NICHE_GENRE",
                "explanation": niche_fallback["explanation"],
                "songs": niche_fallback["songs"],
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)
            return
        
        elif artist_result["status"] != "ARTIST_FOUND":
            st.error(f"Error: {artist_result.get('explanation', 'Unknown error')}")
            return
        
        # Artist found
        artist_name = artist_result["artist"]
        
        # Check duplicate
        if artist_name in st.session_state.excluded_artists:
            st.warning(f"⚠️ {artist_name} already suggested! Searching for alternative...")
            return
        
        st.session_state.excluded_artists.append(artist_name)
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        st.caption(f"Confidence: {artist_result.get('confidence', 'Medium')} | Note: {artist_result.get('note', '')}")
        
        # Step 2: Find and lock official channel (using IMPROVED logic from Code 2)
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Channel")
        
        with st.spinner("Finding official YouTube channel..."):
            locked_channel, channel_status, search_queries = find_and_lock_official_channel(
                artist_name, genre_input, llm
            )
        
        if channel_status == "FOUND" and locked_channel:
            st.success(f"✅ **Artist Channel Locked:** {locked_channel}")
            
            # Step 3: Discover videos (using IMPROVED logic from Code 2)
            with st.spinner(f"Exploring {locked_channel} for music videos..."):
                available_videos = discover_channel_videos(locked_channel, artist_name)
            
            if not available_videos:
                st.error(f"❌ No music videos found in {locked_channel}")
                st.info(f"Channel might not have public music videos.")
                
                session_data = {
                    "genre": genre_input,
                    "artist": artist_name,
                    "locked_channel": locked_channel,
                    "status": "NO_VIDEOS_IN_CHANNEL",
                    "attempt": current_attempt,
                    "timestamp": time.time()
                }
                st.session_state.sessions.append(session_data)
                return
            
            # Select best 3 music videos
            with st.spinner(f"🤖 Using AI to select different songs..."):
                selected_videos = select_best_music_videos(available_videos, 3, llm)
            
            if len(selected_videos) < 3:
                st.warning(f"⚠️ Only found {len(selected_videos)} suitable music videos")
            
            # Display results
            st.markdown("---")
            st.markdown(f"### 🎵 {artist_name} Music Videos from {locked_channel}")
            
            st.success(f"✅ **AI-Selected {len(selected_videos)} Distinct Songs**")
            
            videos_found = 0
            cols = st.columns(3)
            
            for idx, video in enumerate(selected_videos):
                with cols[idx % 3]:
                    st.markdown(f'<div class="video-success">', unsafe_allow_html=True)
                    st.markdown(f"**Song {idx+1}**")
                    
                    # Display video
                    try:
                        st.video(video['url'])
                    except Exception:
                        video_id = video['url'].split('v=')[1].split('&')[0]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                    
                    # Title and info
                    display_title = video['title']
                    if len(display_title) > 40:
                        display_title = display_title[:37] + "..."
                    
                    st.write(f"**{display_title}**")
                    
                    if video.get('duration'):
                        st.caption(f"Duration: {video['duration']}")
                    
                    if video.get('score'):
                        score = video['score']
                        if score >= 70:
                            st.markdown(f'<span class="artist-match-high">Match: {score}/100</span>', unsafe_allow_html=True)
                        elif score >= 40:
                            st.markdown(f'<span class="artist-match-medium">Match: {score}/100</span>', unsafe_allow_html=True)
                    
                    # Watch button
                    st.markdown(
                        f'<a href="{video["url"]}" target="_blank">'
                        '<button style="background-color: #FF0000; color: white; '
                        'border: none; padding: 8px 16px; border-radius: 4px; '
                        'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    videos_found += 1
            
            # Store session
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "locked_channel": locked_channel,
                "status": "COMPLETE",
                "videos_found": videos_found,
                "total_videos": len(selected_videos),
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            
            st.session_state.sessions.append(session_data)
            
            # Summary
            st.markdown("---")
            if videos_found == 3:
                st.success(f"🎉 **Perfect! Found 3 AI-selected songs from {locked_channel}**")
            else:
                st.warning(f"⚠️ Found {videos_found}/3 videos in the channel")
        
        else:
            # No channel found
            st.error(f"❌ Could not find official channel for {artist_name}")
            st.info(f"Try searching '{artist_name}' directly on YouTube.")
            
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "status": "NO_CHANNEL",
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)

if __name__ == "__main__":
    main()
