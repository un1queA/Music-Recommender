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
        # Fallback: compare base titles
        base1 = title1.lower().split('(')[0].split('[')[0].strip()
        base2 = title2.lower().split('(')[0].split('[')[0].strip()
        return base1 != base2

def select_best_music_videos(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """Select the best music videos ensuring unique songs."""
    
    if not videos or not llm:
        return videos[:count] if videos else []
    
    if len(videos) <= count:
        return videos[:count]
    
    sorted_videos = sorted(videos, key=lambda x: x.get('score', 0), reverse=True)
    selected = []
    
    for video in sorted_videos:
        if len(selected) >= count:
            break
        
        is_unique_song = True
        
        if selected:
            for selected_video in selected:
                if not are_songs_different_llm(video['title'], selected_video['title'], llm):
                    is_unique_song = False
                    break
        
        if is_unique_song:
            selected.append(video)
    
    # If still not enough, add remaining high-scored videos
    if len(selected) < count:
        for video in sorted_videos:
            if video not in selected and len(selected) < count:
                selected.append(video)
    
    return selected[:count]

# =============================================================================
# ACCURATE CHANNEL DETECTION WITH SMART OPTIMIZATION
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.5,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1000,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def identify_channel_type_llm(channel_name: str, artist_name: str, video_titles: List[str], llm: ChatOpenAI) -> Dict:
    """
    ACCURATE: Analyze a single channel with multiple video titles for context
    """
    
    video_context = "\n".join([f"- {title}" for title in video_titles[:5]])
    
    prompt = PromptTemplate(
        input_variables=["channel_name", "artist_name", "video_context"],
        template="""Analyze this YouTube channel:

Channel Name: "{channel_name}"
Target Artist: "{artist_name}"

Sample videos from this channel:
{video_context}

Is this the OFFICIAL ARTIST CHANNEL for {artist_name}?

Consider:
1. Does the channel name match the artist name (or close variation)?
2. Are the videos BY this artist (not just featuring them)?
3. Is this a record label/company channel that hosts multiple artists?
4. Is this an auto-generated "Topic" channel?

Return EXACTLY:
TYPE: ARTIST_OFFICIAL or COMPANY or TOPIC or OTHER
CONFIDENCE: High or Medium or Low
REASON: [1-2 sentence explanation]"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "channel_name": channel_name,
            "artist_name": artist_name,
            "video_context": video_context
        })
        
        channel_type = "OTHER"
        confidence = "Low"
        reason = ""
        
        for line in result.strip().split('\n'):
            if 'TYPE:' in line:
                type_text = line.split('TYPE:')[1].strip().upper()
                if 'ARTIST_OFFICIAL' in type_text:
                    channel_type = 'ARTIST_OFFICIAL'
                elif 'COMPANY' in type_text:
                    channel_type = 'COMPANY'
                elif 'TOPIC' in type_text:
                    channel_type = 'TOPIC'
                else:
                    channel_type = 'OTHER'
            elif 'CONFIDENCE:' in line:
                conf_text = line.split('CONFIDENCE:')[1].strip()
                if 'High' in conf_text:
                    confidence = 'High'
                elif 'Medium' in conf_text:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
            elif 'REASON:' in line:
                reason = line.split('REASON:')[1].strip()
        
        return {
            "type": channel_type,
            "confidence": confidence,
            "reason": reason
        }
    except Exception:
        return {
            "type": "OTHER",
            "confidence": "Low",
            "reason": "Error analyzing channel"
        }

def get_artist_search_queries(artist_name: str, genre: str, llm: ChatOpenAI) -> List[str]:
    """Generate optimal search queries."""
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""Generate 3 YouTube search queries to find the OFFICIAL channel for "{artist_name}" ({genre}).

Focus on finding the artist's own channel, not record labels or topic channels.

Return ONLY 3 queries separated by |
Example: BTS official | BTS VEVO | BTS"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist_name": artist_name, "genre": genre})
        queries = [q.strip() for q in result.split('|') if q.strip()]
        return queries[:3] if queries else [artist_name, f"{artist_name} official"]
    except:
        return [artist_name, f"{artist_name} official", f"{artist_name} VEVO"]

def find_and_lock_official_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Tuple[Optional[str], str, List[str]]:
    """
    ACCURATE: Smart pre-filtering + thorough LLM validation
    """
    
    search_queries = get_artist_search_queries(artist_name, genre, llm)
    
    channel_candidates = {}  # channel_name -> {score, videos}
    
    # Gather candidates from searches
    for search_query in search_queries:
        try:
            results = YoutubeSearch(search_query, max_results=15).to_dict()
            
            for result in results:
                channel_name = result['channel'].strip()
                channel_lower = channel_name.lower()
                
                # Skip obvious non-matches
                if 'topic' in channel_lower:
                    continue
                
                # Initialize or update channel data
                if channel_name not in channel_candidates:
                    channel_candidates[channel_name] = {
                        'videos': [],
                        'score': 0
                    }
                
                channel_candidates[channel_name]['videos'].append(result['title'])
                
                # Calculate score
                artist_clean = re.sub(r'[^\w\s]', '', artist_name.lower().strip())
                channel_clean = re.sub(r'[^\w\s]', '', channel_lower.strip())
                
                score = 0
                
                # Exact or very close match
                if artist_clean == channel_clean:
                    score += 150
                elif artist_clean in channel_clean and len(channel_clean) <= len(artist_clean) + 10:
                    score += 100
                elif artist_clean in channel_clean:
                    score += 50
                
                # VEVO is reliable
                if 'vevo' in channel_lower:
                    score += 100
                
                # Official keyword
                if 'official' in channel_lower:
                    score += 30
                
                # Popularity
                if result.get('views'):
                    views = result['views'].lower()
                    if 'm' in views:
                        score += 40
                    elif 'k' in views:
                        score += 20
                
                channel_candidates[channel_name]['score'] = max(
                    channel_candidates[channel_name]['score'],
                    score
                )
            
            time.sleep(0.3)
        except Exception:
            continue
    
    if not channel_candidates:
        return None, "NOT_FOUND", search_queries
    
    # Sort by score and analyze top candidates with LLM
    sorted_channels = sorted(
        channel_candidates.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    # Analyze top 5 candidates individually for accuracy
    for channel_name, data in sorted_channels[:5]:
        if data['score'] < 50:  # Skip low-confidence candidates
            continue
        
        # Use LLM to verify this is the artist's channel
        analysis = identify_channel_type_llm(
            channel_name,
            artist_name,
            data['videos'],
            llm
        )
        
        # Accept if LLM confirms it's the artist's official channel
        if analysis['type'] == 'ARTIST_OFFICIAL':
            # Extra validation: check if channel has enough content
            try:
                verify_search = YoutubeSearch(f"{channel_name}", max_results=10).to_dict()
                channel_videos = [v for v in verify_search if v['channel'].strip() == channel_name]
                
                if len(channel_videos) >= 3:
                    return channel_name, "FOUND", search_queries
            except:
                pass
            
            # If validation fails but LLM is confident, still accept
            if analysis['confidence'] == 'High':
                return channel_name, "FOUND", search_queries
    
    # Fallback: return highest scored if nothing validated
    if sorted_channels:
        return sorted_channels[0][0], "FOUND_UNVERIFIED", search_queries
    
    return None, "NOT_FOUND", search_queries

def is_music_video_llm(video_title: str, artist_name: str, llm: ChatOpenAI) -> bool:
    """
    Determine if a video is a music video.
    """
    
    prompt = PromptTemplate(
        input_variables=["video_title", "artist_name"],
        template="""Is this a MUSIC VIDEO by {artist_name}?

Title: "{video_title}"

Include: official videos, lyric videos, audio, visualizers, music videos
Exclude: interviews, behind-the-scenes, concerts, vlogs, reactions, covers by others

Return ONLY: YES or NO"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"video_title": video_title, "artist_name": artist_name})
        return "YES" in result.strip().upper()
    except Exception:
        # Fallback: basic keyword check
        title_lower = video_title.lower()
        exclude_keywords = ['interview', 'behind', 'concert', 'live', 'vlog', 'reaction', 'cover']
        return not any(kw in title_lower for kw in exclude_keywords)

def discover_channel_videos(locked_channel: str, artist_name: str, llm: ChatOpenAI, min_videos: int = 30) -> List[Dict]:
    """
    Discover and filter music videos from the channel.
    """
    
    if not locked_channel:
        return []
    
    all_videos = []
    
    search_attempts = [
        f"{locked_channel}",
        f"{artist_name} {locked_channel}",
        f"{locked_channel} music video"
    ]
    
    for search_query in search_attempts:
        try:
            results = YoutubeSearch(search_query, max_results=40).to_dict()
            
            for result in results:
                # Must be from our channel
                if result['channel'].strip() != locked_channel:
                    continue
                
                # Basic scoring
                score = score_video_music_likelihood(result, artist_name)
                
                if score >= 30:
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
    
    # Filter with LLM (check top candidates)
    filtered_videos = []
    check_count = 0
    
    for video in sorted(all_videos, key=lambda x: x['score'], reverse=True):
        # Only check high-scored videos with LLM to save time
        if check_count >= 30:  # Limit LLM calls
            break
        
        if is_music_video_llm(video['title'], artist_name, llm):
            filtered_videos.append(video)
        
        check_count += 1
        
        if len(filtered_videos) >= 20:  # Got enough
            break
    
    return sorted(filtered_videos, key=lambda x: x['score'], reverse=True)

def score_video_music_likelihood(video: Dict, artist_name: str) -> int:
    """Score video based on music video characteristics."""
    score = 0
    title = video['title']
    title_lower = title.lower()
    
    # 1. DURATION (important indicator)
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                if 2 <= minutes <= 10:
                    score += 60
                elif 1 <= minutes <= 15:
                    score += 40
        except:
            pass
    
    # 2. TITLE INDICATORS
    music_keywords = ['official', 'video', 'mv', 'audio', 'lyric', 'lyrics', 'visualizer']
    if any(kw in title_lower for kw in music_keywords):
        score += 30
    
    # 3. EXCLUDE NON-MUSIC
    exclude_keywords = ['interview', 'behind', 'making', 'documentary', 'concert', 'live', 'vlog']
    if any(kw in title_lower for kw in exclude_keywords):
        score -= 50
    
    # 4. VIEW COUNT
    if video.get('views'):
        views = video['views'].lower()
        if 'm' in views:
            score += 30
        elif 'k' in views:
            score += 15
    
    # 5. TITLE STRUCTURE
    if re.search(r'[\[\(][^\]\)]+[\]\)]', title):
        score += 10
    
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
        
        if 'POPULAR' in result:
            level = "POPULAR"
            artist_pool = "50+"
        elif 'NICHE' in result:
            level = "NICHE"
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
    
    excluded_text = "None"
    if excluded_artists:
        show_last = min(5, len(excluded_artists))
        excluded_text = ", ".join(excluded_artists[-show_last:])
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text"],
        template="""Find ONE {genre} artist.

Already suggested (don't repeat): {excluded_text}

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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔊 I AM MUSIC")
    st.caption("🎵 AI-Powered Music Discovery - Accurate & Universal")
    
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
            st.error("⚠️ Please enter your DeepSeek API key in the sidebar!")
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
            st.error("❌ Could not find artist. Try another genre.")
            return
        
        artist_name = artist_result["artist"]
        
        if artist_name in st.session_state.excluded_artists:
            st.warning(f"⚠️ {artist_name} already suggested! Try searching again.")
            return
        
        st.session_state.excluded_artists.append(artist_name)
        
        st.success(f"🎤 **Artist Found:** {artist_name}")
        st.caption(f"Confidence: {artist_result.get('confidence', 'Medium')}")
        
        # Find channel
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Channel")
        
        with st.spinner("AI is analyzing channels to find the correct artist channel..."):
            locked_channel, channel_status, _ = find_and_lock_official_channel(
                artist_name, genre_input, llm
            )
        
        if channel_status == "NOT_FOUND" or not locked_channel:
            st.error(f"❌ Could not find official channel for {artist_name}")
            st.info("Try searching the artist directly on YouTube.")
            return
        
        st.success(f"✅ **Official Channel Found:** {locked_channel}")
        
        # Discover videos
        with st.spinner(f"Discovering music videos from {locked_channel}..."):
            available_videos = discover_channel_videos(locked_channel, artist_name, llm)
        
        if not available_videos:
            st.error(f"❌ No music videos found in {locked_channel}")
            st.info("This channel might not have public music videos.")
            return
        
        # Select 3 unique songs
        with st.spinner(f"🤖 Selecting 3 unique songs..."):
            selected_videos = select_best_music_videos(available_videos, 3, llm)
        
        # Display results
        st.markdown("---")
        st.markdown(f"### 🎵 {artist_name} - Music Videos")
        
        if len(selected_videos) < 3:
            st.warning(f"⚠️ Only found {len(selected_videos)} unique song(s)")
        else:
            st.success(f"✅ **Found 3 Unique Songs from {locked_channel}**")
        
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
                if len(display_title) > 45:
                    display_title = display_title[:42] + "..."
                
                st.write(f"**{display_title}**")
                
                if video.get('duration'):
                    st.caption(f"⏱️ {video['duration']}")
                
                if video.get('score'):
                    score = video['score']
                    if score >= 70:
                        st.markdown(f'<span class="artist-match-high">Score: {score}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="artist-match-medium">Score: {score}</span>', unsafe_allow_html=True)
                
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
        
        st.markdown("---")
        st.success(f"✅ Session complete! Search again for more {genre_input} artists.")

if __name__ == "__main__":
    main()
