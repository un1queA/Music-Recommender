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
        
        Are these DIFFERENT SONGS or the SAME SONG (different versions)?
        
        SAME SONG examples:
        - "Dynamite" vs "Dynamite (Official Video)"
        - "Dynamite" vs "Dynamite (Lyrics)"
        - "Dynamite" vs "Dynamite (Audio)"
        - "Dynamite" vs "Dynamite Dance Practice"
        
        Return ONLY: YES or NO
        YES = Different songs
        NO = Same song"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"title1": title1, "title2": title2}).strip().upper()
        return "YES" in result
    except Exception:
        base1 = title1.lower().split('(')[0].split('[')[0].split('-')[0].strip()
        base2 = title2.lower().split('(')[0].split('[')[0].split('-')[0].strip()
        return base1 != base2

def select_best_music_videos(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """Select unique songs - MUST find exactly 'count' unique songs."""
    
    if not videos or not llm:
        return videos[:count] if videos else []
    
    if len(videos) < count:
        return videos  # Return what we have
    
    sorted_videos = sorted(videos, key=lambda x: x.get('score', 0), reverse=True)
    selected = []
    
    # First pass: strict filtering
    for video in sorted_videos:
        if len(selected) >= count:
            break
        
        # Skip obvious non-music
        title_lower = video['title'].lower()
        if any(word in title_lower for word in ['cover', 'dance practice', 'choreography', 'reaction', 'compilation', 'behind']):
            continue
        
        is_unique = True
        if selected:
            # Check against already selected songs
            for sel in selected:
                if not are_songs_different_llm(video['title'], sel['title'], llm):
                    is_unique = False
                    break
        
        if is_unique:
            selected.append(video)
    
    # If we still don't have enough, be less strict
    if len(selected) < count:
        for video in sorted_videos:
            if len(selected) >= count:
                break
            
            # Already selected?
            if video in selected:
                continue
            
            # Check if truly different
            is_unique = True
            for sel in selected:
                if not are_songs_different_llm(video['title'], sel['title'], llm):
                    is_unique = False
                    break
            
            if is_unique:
                selected.append(video)
    
    return selected[:count]

# =============================================================================
# HIGHLY ACCURATE CHANNEL DETECTION
# =============================================================================

def initialize_llm(api_key: str):
    """Initialize the LLM with DeepSeek API"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1200,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek API: {e}")
        return None

def verify_channel_belongs_to_artist(channel_name: str, artist_name: str, llm: ChatOpenAI) -> Dict:
    """
    CRITICAL: Verify that a channel truly belongs to the artist (not a company/label).
    This fetches videos from the channel and checks if they're PRIMARILY by this artist.
    """
    
    try:
        # Get videos from this channel
        channel_videos = YoutubeSearch(f'"{channel_name}"', max_results=20).to_dict()
        
        # Filter to only videos from this exact channel
        channel_videos = [v for v in channel_videos if v['channel'].strip() == channel_name]
        
        if len(channel_videos) < 3:
            return {"belongs_to_artist": False, "reason": "Not enough videos", "confidence": "Low"}
        
        # Get sample titles
        video_titles = [v['title'] for v in channel_videos[:10]]
        titles_text = "\n".join([f"- {title}" for title in video_titles])
        
        prompt = PromptTemplate(
            input_variables=["channel_name", "artist_name", "titles_text"],
            template="""Analyze this YouTube channel:

Channel: "{channel_name}"
Target Artist: "{artist_name}"

Sample videos from this channel:
{titles_text}

CRITICAL QUESTION: Is this the OFFICIAL CHANNEL of {artist_name} where they post THEIR OWN music?

Consider:
1. Are MOST videos by {artist_name} themselves (not other artists)?
2. Is this a record label/company channel that posts multiple artists?
3. Does the channel name match the artist name closely?

Examples:
- "BANGTANTV" is BTS's official channel ✓
- "HYBE LABELS" is a company channel (multiple artists) ✗
- "BLACKPINK" is BLACKPINK's official channel ✓
- "YG Entertainment" is a company channel ✗

Return EXACTLY:
BELONGS_TO_ARTIST: YES or NO
CONFIDENCE: High or Medium or Low
REASON: [brief explanation]"""
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        result = chain.run({
            "channel_name": channel_name,
            "artist_name": artist_name,
            "titles_text": titles_text
        })
        
        belongs = False
        confidence = "Low"
        reason = ""
        
        for line in result.strip().split('\n'):
            if 'BELONGS_TO_ARTIST:' in line:
                belongs = 'YES' in line.upper()
            elif 'CONFIDENCE:' in line:
                if 'High' in line:
                    confidence = 'High'
                elif 'Medium' in line:
                    confidence = 'Medium'
            elif 'REASON:' in line:
                reason = line.split('REASON:')[1].strip()
        
        return {
            "belongs_to_artist": belongs,
            "confidence": confidence,
            "reason": reason,
            "sample_videos": len(channel_videos)
        }
    
    except Exception as e:
        return {"belongs_to_artist": False, "reason": f"Error: {e}", "confidence": "Low"}

def get_artist_search_queries(artist_name: str, genre: str, llm: ChatOpenAI) -> List[str]:
    """Generate optimal search queries."""
    
    prompt = PromptTemplate(
        input_variables=["artist_name", "genre"],
        template="""Generate 3 YouTube search queries to find {artist_name}'s OFFICIAL CHANNEL.

IMPORTANT: We want the artist's OWN channel, not their record label.

Return ONLY 3 queries separated by |
Example: BTS official | BANGTANTV | BTS VEVO"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"artist_name": artist_name, "genre": genre})
        queries = [q.strip() for q in result.split('|') if q.strip()]
        
        # Ensure we have fallbacks
        if not queries:
            queries = [artist_name, f"{artist_name} official"]
        
        # Add VEVO as it's reliable
        if not any('vevo' in q.lower() for q in queries):
            queries.append(f"{artist_name} VEVO")
        
        return queries[:4]
    except:
        return [artist_name, f"{artist_name} official", f"{artist_name} VEVO"]

def find_and_lock_official_channel(artist_name: str, genre: str, llm: ChatOpenAI) -> Tuple[Optional[str], str, List[str]]:
    """
    MOST ACCURATE VERSION: Find the artist's actual channel with thorough validation.
    """
    
    search_queries = get_artist_search_queries(artist_name, genre, llm)
    
    channel_data = {}  # channel_name -> data
    
    # Gather all potential channels
    for search_query in search_queries:
        try:
            results = YoutubeSearch(search_query, max_results=15).to_dict()
            
            for result in results:
                channel_name = result['channel'].strip()
                channel_lower = channel_name.lower()
                
                # Skip obvious rejects
                if 'topic' in channel_lower:
                    continue
                
                if channel_name not in channel_data:
                    channel_data[channel_name] = {
                        'videos': [],
                        'score': 0,
                        'views': []
                    }
                
                channel_data[channel_name]['videos'].append(result['title'])
                if result.get('views'):
                    channel_data[channel_name]['views'].append(result['views'])
                
                # Calculate base score
                artist_clean = re.sub(r'[^\w\s]', '', artist_name.lower().strip())
                channel_clean = re.sub(r'[^\w\s]', '', channel_lower)
                
                score = 0
                
                # Name matching
                if artist_clean == channel_clean:
                    score += 200
                elif artist_clean in channel_clean:
                    # Penalize if channel has many extra words (likely company)
                    words_diff = len(channel_clean.split()) - len(artist_clean.split())
                    if words_diff <= 1:
                        score += 150
                    else:
                        score += 50  # Probably company channel
                
                # VEVO is reliable
                if 'vevo' in channel_lower:
                    score += 120
                
                # Official keyword
                if 'official' in channel_lower and words_diff <= 1:
                    score += 40
                
                # Popularity
                if result.get('views'):
                    views = result['views'].lower()
                    if 'm' in views:
                        score += 30
                
                channel_data[channel_name]['score'] = max(
                    channel_data[channel_name]['score'],
                    score
                )
            
            time.sleep(0.3)
        except Exception:
            continue
    
    if not channel_data:
        return None, "NOT_FOUND", search_queries
    
    # Sort by score
    sorted_channels = sorted(
        channel_data.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    # CRITICAL: Verify each top candidate actually belongs to the artist
    for channel_name, data in sorted_channels[:5]:
        if data['score'] < 80:
            continue
        
        # Use LLM to deeply verify this is the artist's channel
        verification = verify_channel_belongs_to_artist(channel_name, artist_name, llm)
        
        if verification['belongs_to_artist']:
            # Double-check with a final search
            try:
                final_check = YoutubeSearch(f"{channel_name}", max_results=5).to_dict()
                channel_videos = [v for v in final_check if v['channel'].strip() == channel_name]
                
                if len(channel_videos) >= 2:
                    return channel_name, "FOUND", search_queries
            except:
                pass
            
            # If high confidence, accept even without final check
            if verification['confidence'] == 'High':
                return channel_name, "FOUND", search_queries
    
    # No verified channel found
    return None, "NOT_FOUND", search_queries

def is_official_music_video_llm(video_title: str, artist_name: str, llm: ChatOpenAI) -> bool:
    """Check if video is an official music video (not cover, dance practice, etc.)"""
    
    prompt = PromptTemplate(
        input_variables=["video_title", "artist_name"],
        template="""Is this an OFFICIAL MUSIC VIDEO by {artist_name}?

Title: "{video_title}"

INCLUDE: Official music videos, lyric videos, audio, visualizers
EXCLUDE: Covers, dance practice, choreography, behind-the-scenes, interviews, reactions, compilations, remixes by others

Return ONLY: YES or NO"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"video_title": video_title, "artist_name": artist_name})
        return "YES" in result.strip().upper()
    except Exception:
        title_lower = video_title.lower()
        exclude = ['cover', 'dance practice', 'choreography', 'behind', 'interview', 'reaction', 'compilation', 'remix']
        return not any(kw in title_lower for kw in exclude)

def discover_channel_videos(locked_channel: str, artist_name: str, llm: ChatOpenAI) -> List[Dict]:
    """Discover official music videos from the locked channel - AGGRESSIVE MODE."""
    
    if not locked_channel:
        return []
    
    all_videos = []
    
    # More search attempts to find videos
    search_attempts = [
        f'"{locked_channel}"',
        f"{locked_channel} official",
        f"{artist_name} {locked_channel}",
        f"{locked_channel} music video",
        f"{locked_channel} mv"
    ]
    
    for search_query in search_attempts:
        try:
            results = YoutubeSearch(search_query, max_results=50).to_dict()
            
            for result in results:
                # STRICT: Only from locked channel
                if result['channel'].strip() != locked_channel:
                    continue
                
                # Quick filter - only exclude obvious non-music
                title_lower = result['title'].lower()
                if any(word in title_lower for word in ['cover', 'dance practice', 'choreography', 'behind the scenes', 'reaction', 'interview']):
                    continue
                
                score = score_video_music_likelihood(result, artist_name)
                
                # Lower threshold to get more candidates
                if score >= 30:
                    all_videos.append({
                        'url': f"https://www.youtube.com/watch?v={result['id']}",
                        'title': result['title'],
                        'channel': result['channel'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score
                    })
            
            # Keep searching if we don't have enough
            if len(all_videos) >= 40:
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
    
    # LLM filter for quality - check more videos
    filtered = []
    for video in sorted(all_videos, key=lambda x: x['score'], reverse=True)[:30]:
        if is_official_music_video_llm(video['title'], artist_name, llm):
            filtered.append(video)
        
        # Stop once we have enough good candidates
        if len(filtered) >= 20:
            break
    
    return sorted(filtered, key=lambda x: x['score'], reverse=True)

def score_video_music_likelihood(video: Dict, artist_name: str) -> int:
    """Score video as music video."""
    score = 0
    title = video['title'].lower()
    
    # Duration (2-10 min ideal)
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 2 <= mins <= 10:
                    score += 70
                elif 1 <= mins <= 15:
                    score += 40
        except:
            pass
    
    # Music video keywords
    if any(kw in title for kw in ['official', 'video', 'mv', 'audio', 'lyric']):
        score += 40
    
    # Exclude non-music
    if any(kw in title for kw in ['interview', 'behind', 'practice', 'choreography', 'cover', 'reaction']):
        score -= 100
    
    # Views
    if video.get('views'):
        views = video['views'].lower()
        if 'm' in views:
            score += 30
        elif 'k' in views:
            score += 15
    
    return max(0, score)

# =============================================================================
# ARTIST & GENRE HANDLING - FIXED FOR GENRE SWITCHING
# =============================================================================

def analyze_genre_popularity(genre: str, llm: ChatOpenAI) -> Dict:
    """Analyze genre popularity."""
    
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
            "artist_pool": artist_pool
        }
    except:
        return {"level": "MODERATE", "artist_pool": "10-30"}

def discover_artist_with_rotation(genre: str, llm: ChatOpenAI, excluded_for_genre: List[str]) -> Dict:
    """Find artist for specific genre."""
    
    excluded_text = "None"
    if excluded_for_genre:
        excluded_text = ", ".join(excluded_for_genre[-5:])
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded_text"],
        template="""Find ONE {genre} artist.

Already suggested (don't repeat): {excluded_text}

Return ONLY:
ARTIST: [exact name]
CONFIDENCE: High or Medium or Low"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "excluded_text": excluded_text})
        
        artist = ""
        confidence = "Medium"
        
        for line in result.strip().split('\n'):
            if 'ARTIST:' in line:
                artist = line.split('ARTIST:')[1].strip()
            elif 'CONFIDENCE:' in line:
                confidence = line.split('CONFIDENCE:')[1].strip()
        
        if not artist:
            return {"status": "NO_ARTIST_FOUND"}
        
        return {
            "status": "ARTIST_FOUND",
            "artist": artist,
            "confidence": confidence
        }
    except:
        return {"status": "ERROR"}

# =============================================================================
# STREAMLIT APP - FIXED GENRE SWITCHING
# =============================================================================

def main():
    st.set_page_config(
        page_title="IAMUSIC",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize session state - FIXED: Per-genre tracking
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'excluded_by_genre' not in st.session_state:
        st.session_state.excluded_by_genre = {}  # genre -> [artists]
    if 'genre_stats' not in st.session_state:
        st.session_state.genre_stats = {}  # genre -> count
    if 'total_sessions' not in st.session_state:
        st.session_state.total_sessions = 0
    
    # Custom CSS
    st.markdown("""
    <style>
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔊 I AM MUSIC")
    st.caption("🎵 Universal AI Music Discovery - Any Genre, Any Language")
    
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
        st.write(f"**Total searches:** {st.session_state.total_sessions}")
        
        if st.session_state.genre_stats:
            st.markdown("#### By Genre:")
            for genre, count in sorted(st.session_state.genre_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                artists_count = len(st.session_state.excluded_by_genre.get(genre, []))
                st.write(f"**{genre}:** {count} searches, {artists_count} artists")
        
        st.markdown("---")
        
        if st.button("🔄 Clear All Data", type="secondary"):
            st.session_state.excluded_by_genre = {}
            st.session_state.genre_stats = {}
            st.session_state.total_sessions = 0
            st.rerun()
    
    # Main input
    st.markdown("### Enter Any Music Genre")
    
    with st.form(key="search_form"):
        genre_input = st.text_input(
            "Genre:",
            placeholder="e.g., K-pop, J-pop, Tamil Pop, Reggaeton...",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("🔍 Search", use_container_width=True, type="primary")
    
    # Process search with RETRY LOGIC
    if submitted and genre_input:
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your DeepSeek API key!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Get excluded artists for THIS genre only
        genre_key = genre_input.lower().strip()
        if genre_key not in st.session_state.excluded_by_genre:
            st.session_state.excluded_by_genre[genre_key] = []
        
        excluded_for_this_genre = st.session_state.excluded_by_genre[genre_key]
        
        # Track stats
        if genre_key not in st.session_state.genre_stats:
            st.session_state.genre_stats[genre_key] = 0
        st.session_state.genre_stats[genre_key] += 1
        st.session_state.total_sessions += 1
        
        # Analyze genre once
        with st.spinner("Analyzing genre..."):
            genre_analysis = analyze_genre_popularity(genre_input, llm)
        
        # RETRY LOOP: Try up to 5 different artists until we get 3 songs
        MAX_ATTEMPTS = 5
        success = False
        
        status_placeholder = st.empty()
        
        for attempt in range(1, MAX_ATTEMPTS + 1):
            status_placeholder.info(f"🔄 Attempt {attempt}/{MAX_ATTEMPTS}: Finding artist with 3 available songs...")
            
            # Find artist
            artist_result = discover_artist_with_rotation(
                genre_input,
                llm,
                excluded_for_this_genre
            )
            
            if artist_result["status"] != "ARTIST_FOUND":
                if attempt == MAX_ATTEMPTS:
                    status_placeholder.error("❌ Could not find any suitable artists")
                    return
                continue
            
            artist_name = artist_result["artist"]
            
            # Skip if already tried
            if artist_name in excluded_for_this_genre:
                continue
            
            # Try this artist
            status_placeholder.info(f"🎤 Trying artist: {artist_name} (Attempt {attempt}/{MAX_ATTEMPTS})")
            
            # Find channel
            locked_channel, channel_status, _ = find_and_lock_official_channel(
                artist_name, genre_input, llm
            )
            
            if channel_status == "NOT_FOUND" or not locked_channel:
                # Channel not found, try next artist
                st.session_state.excluded_by_genre[genre_key].append(artist_name)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(0.5)
                    continue
                else:
                    status_placeholder.error(f"❌ Could not find valid channels after {MAX_ATTEMPTS} attempts")
                    return
            
            # Discover videos
            available_videos = discover_channel_videos(locked_channel, artist_name, llm)
            
            if not available_videos:
                # No videos found, try next artist
                st.session_state.excluded_by_genre[genre_key].append(artist_name)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(0.5)
                    continue
                else:
                    status_placeholder.error(f"❌ Could not find videos after {MAX_ATTEMPTS} attempts")
                    return
            
            # Select 3 unique songs
            selected_videos = select_best_music_videos(available_videos, 3, llm)
            
            # CRITICAL: Must have exactly 3 songs
            if len(selected_videos) < 3:
                # Not enough songs, try next artist
                st.session_state.excluded_by_genre[genre_key].append(artist_name)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(0.5)
                    continue
                else:
                    status_placeholder.error(f"❌ Could not find 3 unique songs after {MAX_ATTEMPTS} attempts")
                    return
            
            # SUCCESS! We have artist, channel, and 3 songs
            st.session_state.excluded_by_genre[genre_key].append(artist_name)
            success = True
            
            # Clear status and show success
            status_placeholder.empty()
            
            st.markdown('<div class="success-box">🎤 <strong>Artist Found:</strong> ' + artist_name + '</div>', unsafe_allow_html=True)
            st.caption(f"Confidence: {artist_result.get('confidence', 'Medium')}")
            
            st.markdown("---")
            st.markdown('<div class="success-box">✅ <strong>Official Channel:</strong> ' + locked_channel + '</div>', unsafe_allow_html=True)
            
            # Display results
            st.markdown("---")
            st.markdown(f"### 🎵 {artist_name} - Music Videos")
            st.markdown(f'<div class="success-box">✅ Found 3 Unique Songs from {locked_channel}</div>', unsafe_allow_html=True)
            
            cols = st.columns(3)
            
            for idx, video in enumerate(selected_videos):
                with cols[idx % 3]:
                    st.markdown(f"**Song {idx+1}**")
                    
                    try:
                        st.video(video['url'])
                    except Exception:
                        video_id = video['url'].split('v=')[1].split('&')[0]
                        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                    
                    title = video['title']
                    if len(title) > 50:
                        title = title[:47] + "..."
                    
                    st.write(f"**{title}**")
                    
                    if video.get('duration'):
                        st.caption(f"⏱️ {video['duration']}")
                    
                    st.markdown(
                        f'<a href="{video["url"]}" target="_blank">'
                        '<button style="background-color: #FF0000; color: white; '
                        'border: none; padding: 8px 16px; border-radius: 4px; '
                        'cursor: pointer; width: 100%;">▶ Watch on YouTube</button></a>',
                        unsafe_allow_html=True
                    )
            
            st.markdown("---")
            remaining = 50 - len(excluded_for_this_genre)
            if remaining > 0:
                st.success(f"✅ Search again to discover more {genre_input} artists! (~{remaining} more available)")
            else:
                st.info(f"You've explored many {genre_input} artists! Try a different genre.")
            
            break  # Exit retry loop on success
        
        if not success and attempt == MAX_ATTEMPTS:
            status_placeholder.error(f"❌ Could not find a suitable artist with 3 songs after {MAX_ATTEMPTS} attempts. This genre might be too niche or have limited YouTube content.")

if __name__ == "__main__":
    main()
