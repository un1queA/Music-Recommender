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
# LLM-POWERED SONG DEDUPLICATION - IMPROVED
# =============================================================================

def are_songs_different_llm(title1: str, title2: str, llm: ChatOpenAI) -> bool:
    """Use LLM to determine if two videos represent different songs."""
    
    prompt = PromptTemplate(
        input_variables=["title1", "title2"],
        template="""Compare these two video titles:
        Title 1: "{title1}"
        Title 2: "{title2}"
        
        Are these DIFFERENT SONGS or the SAME SONG (different versions/formats)?
        
        Examples of SAME SONG:
        - "Shape of You" vs "Shape of You (Official Video)"
        - "Dynamite" vs "Dynamite (Lyrics)"
        - "Bad Guy" vs "Bad Guy (Audio)"
        
        Return ONLY: YES or NO
        YES = Different songs
        NO = Same song"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"title1": title1, "title2": title2}).strip().upper()
        return "YES" in result
    except Exception:
        return title1.lower().split('(')[0].strip() != title2.lower().split('(')[0].strip()

def select_best_music_videos(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """Select the best music videos ensuring 3 completely different songs."""
    
    if not videos or not llm:
        return videos[:count] if videos else []
    
    if len(videos) <= count:
        return videos[:count]
    
    sorted_videos = sorted(videos, key=lambda x: x.get('score', 0), reverse=True)
    selected = []
    checked_pairs = set()
    
    for video in sorted_videos:
        if len(selected) >= count:
            break
        
        is_unique_song = True
        
        if selected:
            for selected_video in selected:
                # Create a unique pair identifier to avoid re-checking
                pair_id = tuple(sorted([video['title'], selected_video['title']]))
                
                if pair_id in checked_pairs:
                    is_unique_song = False
                    break
                
                # Check if different songs
                if not are_songs_different_llm(video['title'], selected_video['title'], llm):
                    is_unique_song = False
                    checked_pairs.add(pair_id)
                    break
                else:
                    checked_pairs.add(pair_id)
        
        if is_unique_song:
            selected.append(video)
    
    # If still not enough unique songs, relax criteria
    if len(selected) < count and len(sorted_videos) >= count:
        # Add highest scored remaining videos
        for video in sorted_videos:
            if video not in selected and len(selected) < count:
                selected.append(video)
    
    return selected[:count]

# =============================================================================
# OPTIMIZED CHANNEL DETECTION - BATCH LLM CALLS
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,  # Lower for more consistent results
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=800,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def analyze_multiple_channels_llm(candidates: List[Dict], artist_name: str, llm: ChatOpenAI) -> List[Dict]:
    """
    OPTIMIZED: Analyze multiple channels in ONE LLM call instead of multiple calls
    """
    
    if not candidates:
        return []
    
    # Format candidates for batch analysis
    candidates_text = ""
    for idx, c in enumerate(candidates[:10]):  # Limit to top 10
        candidates_text += f"\n{idx+1}. Channel: '{c['channel']}' | Video: '{c['video_title']}'"
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "candidates_text"],
        template="""Analyze these YouTube channels to find the OFFICIAL ARTIST CHANNEL for "{artist_name}":
{candidates_text}

For each channel, determine:
- ARTIST_OFFICIAL: Artist's own channel (e.g., "Taylor Swift", "BTS OFFICIAL")
- COMPANY: Record label/company (e.g., "HYBE LABELS", "Sony Music")  
- TOPIC: Auto-generated topic channel
- OTHER: Fan channel, compilation, etc.

Return in format:
1: TYPE | CONFIDENCE
2: TYPE | CONFIDENCE
...

Example:
1: ARTIST_OFFICIAL | High
2: COMPANY | High
3: TOPIC | Medium"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "artist_name": artist_name,
            "candidates_text": candidates_text
        })
        
        # Parse results
        for idx, line in enumerate(result.strip().split('\n')):
            if ':' in line and idx < len(candidates):
                parts = line.split(':', 1)[1].strip().split('|')
                if len(parts) >= 2:
                    channel_type = parts[0].strip()
                    confidence = parts[1].strip()
                    candidates[idx]['llm_type'] = channel_type
                    candidates[idx]['llm_confidence'] = confidence
        
        return candidates
    
    except Exception:
        # Fallback: mark all as unknown
        for c in candidates:
            c['llm_type'] = "OTHER"
            c['llm_confidence'] = "Low"
        return candidates

def get_artist_search_queries(artist_name: str, genre: str, llm: ChatOpenAI) -> List[str]:
    """Use LLM to generate optimal YouTube search queries"""
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""Generate 3 YouTube search queries to find the OFFICIAL channel for "{artist_name}" ({genre}).

Return ONLY 3 queries separated by |
Example: BTS official | BTS VEVO | BTS"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist_name": artist_name, "genre": genre})
        queries = [q.strip() for q in result.split('|') if q.strip()]
        return queries[:3] if queries else [artist_name]
    except:
        return [artist_name, f"{artist_name} official", f"{artist_name} VEVO"]

def find_and_lock_official_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Tuple[Optional[str], str, List[str]]:
    """
    OPTIMIZED: Batch process channel analysis
    """
    
    # Get search queries
    search_queries = get_artist_search_queries(artist_name, genre, llm)
    
    all_candidates = []
    seen_channels = set()
    
    for search_query in search_queries[:2]:  # Limit to 2 searches for speed
        try:
            results = YoutubeSearch(search_query, max_results=10).to_dict()
            
            for result in results:
                channel_name = result['channel'].strip()
                
                # Skip duplicates
                if channel_name in seen_channels:
                    continue
                seen_channels.add(channel_name)
                
                # Basic filtering
                channel_lower = channel_name.lower()
                if 'topic' in channel_lower:
                    continue
                
                # Calculate base score
                score = 0
                
                # Name similarity
                artist_clean = re.sub(r'[^\w\s]', '', artist_name.lower().strip())
                channel_clean = re.sub(r'[^\w\s]', '', channel_lower.strip())
                
                if artist_clean == channel_clean:
                    score += 100
                elif artist_clean in channel_clean:
                    score += 50
                
                # VEVO bonus
                if 'vevo' in channel_lower:
                    score += 80
                
                # View count
                if result.get('views'):
                    views_text = result['views'].lower()
                    if 'm' in views_text:
                        score += 40
                    elif 'k' in views_text:
                        score += 20
                
                if score >= 50:
                    all_candidates.append({
                        'channel': channel_name,
                        'score': score,
                        'video_title': result['title'],
                        'views': result.get('views', '')
                    })
            
            time.sleep(0.2)
        except Exception:
            continue
    
    if not all_candidates:
        return None, "NOT_FOUND", search_queries
    
    # BATCH ANALYZE with LLM (one call for all candidates)
    all_candidates = analyze_multiple_channels_llm(all_candidates, artist_name, llm)
    
    # Select best ARTIST_OFFICIAL channel
    official_channels = [
        c for c in all_candidates 
        if c.get('llm_type') == 'ARTIST_OFFICIAL'
    ]
    
    if official_channels:
        # Sort by score and confidence
        official_channels.sort(
            key=lambda x: (
                1 if x.get('llm_confidence') == 'High' else 0,
                x['score']
            ),
            reverse=True
        )
        best = official_channels[0]
        return best['channel'], "FOUND", search_queries
    
    # Fallback to highest scored
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates[0]['channel'], "FOUND_UNVERIFIED", search_queries

def batch_filter_music_videos_llm(videos: List[Dict], artist_name: str, llm: ChatOpenAI) -> List[Dict]:
    """
    OPTIMIZED: Filter multiple videos in ONE LLM call
    """
    
    if not videos:
        return []
    
    # Format videos for batch analysis
    videos_text = ""
    for idx, v in enumerate(videos[:20]):  # Limit to 20
        videos_text += f"\n{idx+1}. {v['title']}"
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "videos_text"],
        template="""Which of these are MUSIC VIDEOS by {artist_name}?
        
Exclude: interviews, behind-the-scenes, concerts, vlogs, tutorials
Include: official videos, lyric videos, audio, visualizers
{videos_text}

Return ONLY the numbers of music videos, comma-separated:
Example: 1,3,5,7,9"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "artist_name": artist_name,
            "videos_text": videos_text
        })
        
        # Parse numbers
        music_video_indices = set()
        for part in result.strip().split(','):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(videos):
                    music_video_indices.add(idx)
            except:
                continue
        
        # Return filtered videos
        return [videos[i] for i in sorted(music_video_indices)]
    
    except Exception:
        # Fallback: return all
        return videos

def discover_channel_videos(locked_channel: str, artist_name: str, llm: ChatOpenAI, min_videos: int = 20) -> List[Dict]:
    """
    OPTIMIZED: Batch filter videos with LLM
    """
    
    if not locked_channel:
        return []
    
    all_videos = []
    
    search_attempts = [
        f"{locked_channel}",
        f"{artist_name} {locked_channel}"
    ]
    
    for search_query in search_attempts:
        try:
            results = YoutubeSearch(search_query, max_results=30).to_dict()
            
            for result in results:
                # Only from our channel
                if result['channel'].strip() != locked_channel:
                    continue
                
                score = score_video_music_likelihood(result, artist_name)
                
                if score >= 20:  # Lower threshold, let LLM do heavy filtering
                    all_videos.append({
                        'url': f"https://www.youtube.com/watch?v={result['id']}",
                        'title': result['title'],
                        'channel': result['channel'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score
                    })
            
            if len(all_videos) >= min_videos:
                break
                
        except Exception:
            continue
    
    # Remove duplicates
    unique_videos = {}
    for video in all_videos:
        video_id = video['url'].split('v=')[1].split('&')[0]
        if video_id not in unique_videos:
            unique_videos[video_id] = video
    
    all_videos = list(unique_videos.values())
    
    # BATCH FILTER with LLM (one call for all videos)
    if all_videos:
        all_videos = batch_filter_music_videos_llm(all_videos, artist_name, llm)
    
    return sorted(all_videos, key=lambda x: x['score'], reverse=True)

def score_video_music_likelihood(video: Dict, artist_name: str) -> int:
    """Score video based on music video characteristics."""
    score = 0
    title = video['title']
    
    # 1. DURATION
    if video.get('duration'):
        duration = video['duration']
        try:
            parts = duration.split(':')
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), int(parts[1])
                total_seconds = minutes * 60 + seconds
                
                if 120 <= total_seconds <= 600:
                    score += 60
                elif 90 <= total_seconds <= 720:
                    score += 40
                elif 30 <= total_seconds <= 1200:
                    score += 20
        except:
            pass
    
    # 2. TITLE PATTERNS
    title_lower = title.lower()
    
    if re.search(r'[\[\(][^\]\)]+[\]\)]', title):
        score += 10
    
    # 3. VIEW POPULARITY
    if video.get('views'):
        views_text = video['views'].lower()
        if 'm' in views_text:
            score += 25
        elif 'k' in views_text:
            score += 15
    
    # 4. FORMAT HINTS
    format_indicators = ['official', 'video', 'mv', 'lyric', 'audio']
    if any(ind in title_lower for ind in format_indicators):
        score += 15
    
    return max(0, score)

# =============================================================================
# ARTIST & GENRE HANDLING
# =============================================================================

def analyze_genre_popularity(genre: str, llm: ChatOpenAI, search_attempts: int) -> Dict:
    """Analyze genre for popularity."""
    
    prompt = PromptTemplate(
        input_variables=["genre"],
        template="""Analyze music genre: "{genre}"
        
Return ONLY:
LEVEL: POPULAR or MODERATE or NICHE
POOL: 50+ or 10-30 or 1-10"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre})
        
        level = "MODERATE"
        artist_pool = "10-30"
        
        for line in result.strip().split('\n'):
            if 'LEVEL:' in line:
                if 'POPULAR' in line:
                    level = "POPULAR"
                elif 'NICHE' in line:
                    level = "NICHE"
            elif 'POOL:' in line:
                if '50+' in line:
                    artist_pool = "50+"
                elif '1-10' in line:
                    artist_pool = "1-10"
        
        return {
            "level": level,
            "artist_pool": artist_pool,
            "attempts": search_attempts
        }
    except Exception:
        return {
            "level": "MODERATE",
            "artist_pool": "10-30",
            "attempts": search_attempts
        }

def discover_artist_with_rotation(genre: str, llm: ChatOpenAI, excluded_artists: List[str], 
                                 genre_popularity: Dict) -> Dict:
    """Find artist with intelligent rotation."""
    
    current_attempt = genre_popularity["attempts"]
    
    # Format excluded artists
    excluded_text = "None"
    if excluded_artists:
        show_last = min(5, len(excluded_artists))
        excluded_text = ", ".join(excluded_artists[-show_last:])
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text"],
        template="""Find ONE {genre} artist.

Excluded: {excluded_text}

Return ONLY:
ARTIST: [name]
CONFIDENCE: High or Medium or Low"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "genre": genre,
            "excluded_text": excluded_text
        })
        
        artist = ""
        confidence = "Medium"
        
        for line in result.strip().split('\n'):
            if 'ARTIST:' in line:
                artist = line.split('ARTIST:')[1].strip()
            elif 'CONFIDENCE:' in line:
                confidence = line.split('CONFIDENCE:')[1].strip()
        
        if not artist:
            return {
                "status": "NO_ARTIST_FOUND",
                "attempts": current_attempt
            }
        
        return {
            "status": "ARTIST_FOUND",
            "artist": artist,
            "confidence": confidence,
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
    
    return {
        "explanation": f"'{genre}' is too niche for YouTube official channels.",
        "songs": [f"{genre} song 1", f"{genre} song 2", f"{genre} song 3"],
        "suggestions": f"Try searching '{genre}' on Bandcamp or SoundCloud.",
        "genre": genre,
        "attempts": attempts
    }

# =============================================================================
# STREAMLIT APP
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
    
    # Custom CSS
    st.markdown("""
    <style>
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
    
    # Header
    st.title("🔊 I AM MUSIC")
    st.caption("🚀 Fast AI-Powered Music Discovery - Works with any language!")
    
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
        
        if st.button("🔄 Clear All Data", type="secondary"):
            st.session_state.sessions = []
            st.session_state.excluded_artists = []
            st.session_state.genre_attempts = {}
            st.session_state.genre_popularity = {}
            st.rerun()
    
    # Main input
    st.markdown("### Enter Any Music Genre")
    
    with st.form(key="search_form"):
        genre_input = st.text_input(
            "Genre name:",
            placeholder="e.g., Tamil Pop, K-pop, Reggaeton, Synthwave...",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("🔍 Search", use_container_width=True, type="primary")
    
    # Process search
    if submitted and genre_input:
        if not st.session_state.api_key:
            st.error("Please enter your DeepSeek API key!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Track attempts
        if genre_input not in st.session_state.genre_attempts:
            st.session_state.genre_attempts[genre_input] = 0
        st.session_state.genre_attempts[genre_input] += 1
        current_attempt = st.session_state.genre_attempts[genre_input]
        
        # Analyze genre
        if genre_input not in st.session_state.genre_popularity:
            with st.spinner("Analyzing genre..."):
                genre_analysis = analyze_genre_popularity(genre_input, llm, current_attempt)
                st.session_state.genre_popularity[genre_input] = genre_analysis
        else:
            genre_analysis = st.session_state.genre_popularity[genre_input]
            genre_analysis["attempts"] = current_attempt
        
        # Find artist
        with st.spinner(f"Finding {genre_input} artist..."):
            artist_result = discover_artist_with_rotation(
                genre_input,
                llm,
                st.session_state.excluded_artists,
                genre_analysis
            )
        
        if artist_result["status"] != "ARTIST_FOUND":
            st.error("Could not find artist. Try another genre.")
            return
        
        artist_name = artist_result["artist"]
        
        if artist_name in st.session_state.excluded_artists:
            st.warning(f"⚠️ {artist_name} already suggested!")
            return
        
        st.session_state.excluded_artists.append(artist_name)
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        
        # Find channel
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Channel")
        
        with st.spinner("Finding official channel..."):
            locked_channel, channel_status, _ = find_and_lock_official_channel(
                artist_name, genre_input, llm
            )
        
        if channel_status == "NOT_FOUND" or not locked_channel:
            st.error(f"❌ Could not find official channel for {artist_name}")
            return
        
        st.success(f"✅ **Channel Locked:** {locked_channel}")
        
        # Discover videos
        with st.spinner(f"Finding music videos..."):
            available_videos = discover_channel_videos(locked_channel, artist_name, llm)
        
        if not available_videos:
            st.error(f"❌ No music videos found")
            return
        
        # Select 3 unique songs
        with st.spinner(f"🤖 Selecting 3 unique songs..."):
            selected_videos = select_best_music_videos(available_videos, 3, llm)
        
        # Display results
        st.markdown("---")
        st.markdown(f"### 🎵 {artist_name} - 3 Unique Songs")
        
        if len(selected_videos) < 3:
            st.warning(f"⚠️ Only found {len(selected_videos)} unique songs")
        else:
            st.success(f"✅ **Found 3 Different Songs**")
        
        cols = st.columns(3)
        
        for idx, video in enumerate(selected_videos):
            with cols[idx % 3]:
                st.markdown(f"**Song {idx+1}**")
                
                try:
                    st.video(video['url'])
                except Exception:
                    video_id = video['url'].split('v=')[1].split('&')[0]
                    st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                
                display_title = video['title']
                if len(display_title) > 40:
                    display_title = display_title[:37] + "..."
                
                st.write(f"**{display_title}**")
                
                if video.get('duration'):
                    st.caption(f"⏱️ {video['duration']}")
                
                st.markdown(
                    f'<a href="{video["url"]}" target="_blank">'
                    '<button style="background-color: #FF0000; color: white; '
                    'border: none; padding: 8px 16px; border-radius: 4px; '
                    'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                    unsafe_allow_html=True
                )
        
        # Store session
        st.session_state.sessions.append({
            "genre": genre_input,
            "artist": artist_name,
            "locked_channel": locked_channel,
            "status": "COMPLETE",
            "videos_found": len(selected_videos),
            "timestamp": time.time()
        })

if __name__ == "__main__":
    main()
