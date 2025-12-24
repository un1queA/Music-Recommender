# prerequisites:
# pip install streamlit langchain-openai langchain_core google-api-python-client isodate
# Run: streamlit run app.py

import streamlit as st
import re
import time
import isodate
from typing import Dict, List, Optional, Tuple

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import YouTube Data API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# YOUTUBE DATA API CLIENT SETUP
# =============================================================================
def get_youtube_client(api_key: str):
    """Initialize the YouTube Data API client."""
    if not api_key:
        return None
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        return youtube
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {e}")
        return None

# =============================================================================
# OFFICIAL CHANNEL FINDING USING YOUTUBE DATA API
# =============================================================================
def find_artist_channel_api(artist_name: str, youtube) -> Tuple[Optional[Dict], str]:
    """
    Find artist's channel using YouTube Data API with OAC heuristics.
    Returns (channel_info, status)
    """
    if not youtube:
        return None, "API_NOT_INITIALIZED"
    
    try:
        # Search for channels with the artist name
        search_response = youtube.search().list(
            q=f"{artist_name} official artist channel",
            part="snippet",
            type="channel",
            maxResults=15,
            safeSearch="none"
        ).execute()
        
        channel_candidates = []
        
        for item in search_response.get('items', []):
            channel_id = item['id']['channelId']
            channel_title = item['snippet']['title']
            
            # Get detailed channel information
            channel_response = youtube.channels().list(
                id=channel_id,
                part="snippet,statistics,contentDetails,brandingSettings"
            ).execute()
            
            if not channel_response.get('items'):
                continue
                
            channel_info = channel_response['items'][0]
            snippet = channel_info.get('snippet', {})
            stats = channel_info.get('statistics', {})
            branding = channel_info.get('brandingSettings', {})
            
            # Calculate OAC score based on available API data
            score = 0
            indicators = []
            
            # 1. Channel name matches artist (highest priority)
            artist_lower = artist_name.lower()
            channel_title_lower = channel_title.lower()
            
            if artist_lower == channel_title_lower:
                score += 100
                indicators.append("Exact artist name match")
            elif artist_lower in channel_title_lower:
                score += 70
                indicators.append("Partial artist name match")
            
            # 2. Channel has "Official" in title
            if 'official' in channel_title_lower:
                score += 40
                indicators.append("'Official' in channel title")
            
            # 3. Check for music-related keywords in description
            description = snippet.get('description', '').lower()
            music_keywords = ['music', 'artist', 'band', 'singer', 'musician', 'record', 'label']
            for keyword in music_keywords:
                if keyword in description:
                    score += 20
                    indicators.append(f"Music-related: '{keyword}'")
                    break
            
            # 4. Subscriber count (established channels)
            subscriber_count = int(stats.get('subscriberCount', 0))
            if subscriber_count > 1000000:  # 1M+ subscribers
                score += 60
                indicators.append("Large subscriber base (1M+)")
            elif subscriber_count > 100000:  # 100K+ subscribers
                score += 40
                indicators.append("Established subscriber base (100K+)")
            elif subscriber_count > 10000:  # 10K+ subscribers
                score += 20
                indicators.append("Growing subscriber base (10K+)")
            
            # 5. Video count (active channel)
            video_count = int(stats.get('videoCount', 0))
            if video_count > 100:
                score += 30
                indicators.append("Active channel (100+ videos)")
            elif video_count > 20:
                score += 15
                indicators.append("Moderately active channel (20+ videos)")
            
            # 6. Custom URL (often indicates official channel)
            if snippet.get('customUrl'):
                score += 25
                indicators.append("Has custom URL")
            
            # 7. Check for topic channels (often auto-generated)
            if 'topic' in channel_title_lower or ' - topic' in channel_title_lower:
                score -= 50  # Penalize topic channels
                indicators.append("Topic channel (penalty)")
            
            # Only consider promising candidates
            if score >= 50:
                channel_data = {
                    'id': channel_id,
                    'title': channel_title,
                    'description': snippet.get('description', ''),
                    'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                    'subscribers': subscriber_count,
                    'videos': video_count,
                    'views': int(stats.get('viewCount', 0)),
                    'score': score,
                    'indicators': indicators,
                    'upload_playlist_id': channel_info.get('contentDetails', {}).get('relatedPlaylists', {}).get('uploads', '')
                }
                channel_candidates.append(channel_data)
        
        # Select best candidate
        if channel_candidates:
            channel_candidates.sort(key=lambda x: x['score'], reverse=True)
            return channel_candidates[0], "CHANNEL_FOUND"
        
        return None, "NO_CHANNEL_FOUND"
        
    except HttpError as e:
        if e.resp.status == 403:
            return None, "QUOTA_EXCEEDED"
        return None, f"API_ERROR: {str(e)}"
    except Exception as e:
        return None, f"ERROR: {str(e)}"

# =============================================================================
# VIDEO DISCOVERY WITH GUARANTEED 3 VIDEOS
# =============================================================================
def get_channel_videos_api(channel_id: str, upload_playlist_id: str, youtube, max_videos: int = 50) -> List[Dict]:
    """Get videos from a channel using YouTube Data API."""
    videos = []
    
    if not youtube or not upload_playlist_id:
        return videos
    
    try:
        # Get videos from uploads playlist
        playlist_items = youtube.playlistItems().list(
            playlistId=upload_playlist_id,
            part="snippet,contentDetails",
            maxResults=min(max_videos, 50)
        ).execute()
        
        video_ids = [item['contentDetails']['videoId'] for item in playlist_items.get('items', [])]
        
        if video_ids:
            # Get detailed video information
            videos_response = youtube.videos().list(
                id=','.join(video_ids),
                part="snippet,contentDetails,statistics",
                maxResults=len(video_ids)
            ).execute()
            
            for item in videos_response.get('items', []):
                # Calculate video score for music likelihood
                score = score_video_for_music_api(item)
                
                videos.append({
                    'id': item['id'],
                    'url': f"https://www.youtube.com/watch?v={item['id']}",
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'channel': item['snippet']['channelTitle'],
                    'published': item['snippet']['publishedAt'],
                    'duration': item['contentDetails']['duration'],
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)),
                    'score': score,
                    'thumbnail': item['snippet']['thumbnails']['high']['url']
                })
        
        return videos
        
    except Exception as e:
        return videos

def score_video_for_music_api(video: Dict) -> int:
    """Score video for music likelihood using API data."""
    score = 0
    title = video['snippet']['title'].lower()
    
    # 1. Official markers
    official_indicators = ['official music video', 'official video', 'official audio', 'mv official']
    for indicator in official_indicators:
        if indicator in title:
            score += 60
            break
    
    # 2. Music video indicators
    video_indicators = ['music video', ' mv ', 'lyric video', 'audio visualizer']
    for indicator in video_indicators:
        if indicator in title:
            score += 40
            break
    
    # 3. Duration analysis
    try:
        duration_seconds = isodate.parse_duration(video['contentDetails']['duration']).total_seconds()
        if 120 <= duration_seconds <= 600:  # 2-10 minutes
            score += 40
        elif 90 <= duration_seconds <= 720:  # 1.5-12 minutes
            score += 25
    except:
        pass
    
    # 4. View count
    views = int(video['statistics'].get('viewCount', 0))
    if views > 1000000:
        score += 35
    elif views > 100000:
        score += 20
    elif views > 10000:
        score += 10
    
    # 5. Recent content (within 2 years)
    try:
        from datetime import datetime, timezone
        published = datetime.fromisoformat(video['snippet']['publishedAt'].replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - published).days
        if age_days < 365 * 2:  # Less than 2 years
            score += 20
    except:
        pass
    
    # 6. Avoid non-music content
    non_music = ['interview', 'behind the scenes', 'making of', 'documentary', 'cover', 'live at']
    for indicator in non_music:
        if indicator in title:
            score -= 30
            break
    
    return max(0, score)

def ensure_three_videos_api(artist_name: str, channel_info: Dict, youtube, llm: ChatOpenAI) -> List[Dict]:
    """
    GUARANTEE 3 videos using multiple strategies.
    """
    all_videos = []
    
    # Strategy 1: Get videos from the found channel
    if channel_info and 'upload_playlist_id' in channel_info:
        channel_videos = get_channel_videos_api(
            channel_info['id'], 
            channel_info['upload_playlist_id'], 
            youtube,
            max_videos=50
        )
        if channel_videos:
            # Try to get distinct songs using LLM
            distinct_videos = select_best_music_videos(channel_videos, 3, llm)
            all_videos.extend(distinct_videos)
    
    # Strategy 2: Search for artist videos if we don't have 3
    if len(all_videos) < 3 and youtube:
        try:
            search_response = youtube.search().list(
                q=f"{artist_name} official music video",
                part="snippet",
                type="video",
                maxResults=30,
                videoDuration="medium",  # 4-20 minutes
                order="relevance"
            ).execute()
            
            search_videos = []
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                
                # Get video details
                video_response = youtube.videos().list(
                    id=video_id,
                    part="snippet,contentDetails,statistics"
                ).execute()
                
                if video_response.get('items'):
                    video_data = video_response['items'][0]
                    score = score_video_for_music_api(video_data)
                    
                    search_videos.append({
                        'id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': video_data['snippet']['title'],
                        'channel': video_data['snippet']['channelTitle'],
                        'duration': video_data['contentDetails']['duration'],
                        'views': int(video_data['statistics'].get('viewCount', 0)),
                        'score': score,
                        'thumbnail': video_data['snippet']['thumbnails']['high']['url']
                    })
            
            if search_videos:
                search_videos.sort(key=lambda x: x['score'], reverse=True)
                for video in search_videos:
                    if len(all_videos) >= 3:
                        break
                    
                    # Check if different from existing videos
                    is_different = True
                    for existing in all_videos:
                        if not are_songs_different_llm(video['title'], existing['title'], llm):
                            is_different = False
                            break
                    
                    if is_different:
                        all_videos.append(video)
                        
        except Exception:
            pass
    
    # Strategy 3: Generate search links as ultimate fallback
    if len(all_videos) < 3:
        fallback_links = [
            {
                'url': f"https://www.youtube.com/results?search_query={artist_name}+official+music+video",
                'title': f"{artist_name} - Official Music Videos",
                'channel': "YouTube Search",
                'duration': "Various",
                'views': "Search results",
                'score': 50,
                'is_fallback': True,
                'thumbnail': "https://img.youtube.com/vi/dummy/hqdefault.jpg"
            },
            {
                'url': f"https://www.youtube.com/results?search_query={artist_name}+new+song",
                'title': f"{artist_name} - Latest Release",
                'channel': "YouTube Search",
                'duration': "Various",
                'views': "Search results",
                'score': 50,
                'is_fallback': True,
                'thumbnail': "https://img.youtube.com/vi/dummy/hqdefault.jpg"
            },
            {
                'url': f"https://www.youtube.com/results?search_query={artist_name}+top+songs",
                'title': f"{artist_name} - Popular Songs",
                'channel': "YouTube Search",
                'duration': "Various",
                'views': "Search results",
                'score': 50,
                'is_fallback': True,
                'thumbnail': "https://img.youtube.com/vi/dummy/hqdefault.jpg"
            }
        ]
        
        for i in range(3 - len(all_videos)):
            if i < len(fallback_links):
                all_videos.append(fallback_links[i])
    
    return all_videos[:3]

# =============================================================================
# LLM FUNCTIONS (Keep from your original code)
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

def initialize_llm(api_key: str):
    """Initialize the DeepSeek LLM."""
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
# ARTIST & GENRE HANDLING (Keep from your original code)
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
# STREAMLIT APP (YOUR ORIGINAL UI - UNCHANGED)
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
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {"youtube": "", "deepseek": ""}
    if 'excluded_artists' not in st.session_state:
        st.session_state.excluded_artists = []
    if 'genre_attempts' not in st.session_state:
        st.session_state.genre_attempts = {}
    if 'genre_popularity' not in st.session_state:
        st.session_state.genre_popularity = {}
    
    # Custom CSS (your original CSS)
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
    
    .video-success {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 15px;
    }
    
    .oac-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        display: inline-block;
        margin: 5px;
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
    
    .guarantee-badge {
        background: linear-gradient(135deg, #FF5722 0%, #FF9800 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 1em;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
        border: 2px solid #FF5722;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header (your original header)
    st.title("🔊 I AM MUSIC")
    st.markdown('<div class="guarantee-badge">🎯 GUARANTEED 3 MUSIC VIDEOS</div>', unsafe_allow_html=True)
    
    # Sidebar (your original sidebar with added YouTube API key)
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # YouTube API Key
        youtube_api_key = st.text_input(
            "YouTube Data API Key:",
            type="password",
            value=st.session_state.api_keys["youtube"],
            placeholder="AIzaSy... (from Google Cloud Console)"
        )
        
        # DeepSeek API Key
        deepseek_api_key = st.text_input(
            "DeepSeek API Key:",
            type="password",
            value=st.session_state.api_keys["deepseek"],
            placeholder="sk-..."
        )
        
        if youtube_api_key:
            st.session_state.api_keys["youtube"] = youtube_api_key
        if deepseek_api_key:
            st.session_state.api_keys["deepseek"] = deepseek_api_key
        
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
            st.session_state.api_keys = {"youtube": "", "deepseek": ""}
            st.rerun()
    
    # Main input (your original UI)
    st.markdown("### Enter Any Music Genre")
    
    col1, col2, col3 = st.columns([2,1,1])
    
    with col1:
        with st.form(key="search_form"):
            genre_input = st.text_input(
                "Genre name:",
                placeholder="e.g., K-pop, Reggaeton, Synthwave, Tamil Pop...",
                label_visibility="collapsed",
                key="genre_input"
            )
            
            submitted = st.form_submit_button("Search", use_container_width=True, type="primary")
    
    # Process search
    if submitted and genre_input:
        # Check for API keys
        if not st.session_state.api_keys["youtube"]:
            st.error("Please enter your YouTube Data API key!")
            return
        if not st.session_state.api_keys["deepseek"]:
            st.error("Please enter your DeepSeek API key!")
            return
        
        # Initialize APIs
        youtube = get_youtube_client(st.session_state.api_keys["youtube"])
        llm = initialize_llm(st.session_state.api_keys["deepseek"])
        
        if not youtube or not llm:
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
        
        # Step 2: Find and lock official channel using YouTube Data API
        st.markdown("---")
        st.markdown("### 🔍 Finding Official Artist Channel")
        
        with st.spinner("Searching for official YouTube channel using YouTube Data API..."):
            channel_info, channel_status = find_artist_channel_api(artist_name, youtube)
        
        if channel_status == "CHANNEL_FOUND" and channel_info:
            st.success(f"✅ **Artist Channel Found:** {channel_info['title']}")
            
            # Display channel info
            with st.expander("Channel Analysis"):
                st.write(f"**Channel:** {channel_info['title']}")
                st.write(f"**Subscribers:** {channel_info['subscribers']:,}")
                st.write(f"**Videos:** {channel_info['videos']}")
                st.write(f"**Confidence Score:** {channel_info['score']}/100")
                
                if channel_info.get('indicators'):
                    st.write("**Detection Indicators:**")
                    for indicator in channel_info['indicators'][:5]:
                        st.write(f"• {indicator}")
            
            # Step 3: Discover music videos with GUARANTEE of 3 videos
            st.markdown("---")
            st.markdown("### 🎵 Discovering Music Videos")
            
            with st.spinner(f"🔍 **GUARANTEE:** Finding 3 distinct music videos..."):
                selected_videos = ensure_three_videos_api(artist_name, channel_info, youtube, llm)
            
            # Display results
            st.markdown("---")
            st.markdown(f"### 🎬 {artist_name} Music Videos")
            
            st.success(f"✅ **SUCCESS:** Found {len(selected_videos)} music videos")
            
            # Display 3 videos in columns
            if len(selected_videos) >= 3:
                cols = st.columns(3)
                
                for idx, video in enumerate(selected_videos[:3]):
                    with cols[idx]:
                        st.markdown(f'<div class="video-success">', unsafe_allow_html=True)
                        
                        # Display fallback notice if needed
                        if video.get('is_fallback'):
                            st.warning("⚠️ Search Link")
                        
                        st.markdown(f"**Song {idx+1}**")
                        
                        # Display thumbnail
                        try:
                            st.image(video['thumbnail'], use_column_width=True)
                        except:
                            st.image("https://img.youtube.com/vi/dummy/hqdefault.jpg", use_column_width=True)
                        
                        # Title and info
                        display_title = video['title']
                        if len(display_title) > 50:
                            display_title = display_title[:47] + "..."
                        
                        st.write(f"**{display_title}**")
                        
                        # Video info
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            if video.get('duration'):
                                # Convert ISO duration to readable format
                                try:
                                    duration_seconds = isodate.parse_duration(video['duration']).total_seconds()
                                    minutes = int(duration_seconds // 60)
                                    seconds = int(duration_seconds % 60)
                                    st.caption(f"⏱️ {minutes}:{seconds:02d}")
                                except:
                                    st.caption(f"⏱️ {video['duration']}")
                            elif 'duration' in video:
                                st.caption(f"⏱️ {video['duration']}")
                        with col_info2:
                            if video.get('views'):
                                if isinstance(video['views'], int):
                                    if video['views'] > 1000000:
                                        st.caption(f"👁️ {video['views']//1000000}M views")
                                    elif video['views'] > 1000:
                                        st.caption(f"👁️ {video['views']//1000}K views")
                                    else:
                                        st.caption(f"👁️ {video['views']} views")
                                else:
                                    st.caption(f"👁️ {video['views']}")
                        
                        # Quality score
                        if video.get('score') and not video.get('is_fallback'):
                            score = video['score']
                            if score >= 70:
                                st.markdown(f'<span class="artist-match-high">Quality: {score}/100</span>', unsafe_allow_html=True)
                            elif score >= 40:
                                st.markdown(f'<span class="artist-match-medium">Quality: {score}/100</span>', unsafe_allow_html=True)
                        
                        # Watch button
                        button_text = "🔍 Search on YouTube" if video.get('is_fallback') else "▶ Watch on YouTube"
                        button_color = "#4285F4" if video.get('is_fallback') else "#FF0000"
                        st.markdown(
                            f'<a href="{video["url"]}" target="_blank">'
                            f'<button style="background-color: {button_color}; color: white; '
                            f'border: none; padding: 8px 16px; border-radius: 4px; '
                            f'cursor: pointer; width: 100%; margin-top: 10px;">{button_text}</button></a>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Store session
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "channel": channel_info['title'] if channel_info else None,
                "channel_status": channel_status,
                "status": "COMPLETE",
                "videos_found": len(selected_videos),
                "total_videos": 3,
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            
            st.session_state.sessions.append(session_data)
            
            # Summary
            st.markdown("---")
            st.success(f"🎉 **COMPLETE:** Successfully provided 3 music videos for {artist_name}")
            
            if any(v.get('is_fallback') for v in selected_videos):
                st.info("ℹ️ Some search links were provided when direct videos couldn't be found.")
        
        else:
            # No channel found - try to get videos anyway
            st.error(f"❌ Could not find official channel for {artist_name}")
            
            if channel_status == "QUOTA_EXCEEDED":
                st.warning("YouTube API quota may be exceeded. Try again tomorrow.")
            
            # Try to get videos without channel
            with st.spinner("Trying to find music videos without channel lock..."):
                fallback_videos = ensure_three_videos_api(artist_name, None, youtube, llm)
            
            if fallback_videos:
                st.warning(f"⚠️ Found {len(fallback_videos)} videos (no channel lock)")
                
                # Display fallback videos
                cols = st.columns(3)
                for idx, video in enumerate(fallback_videos[:3]):
                    with cols[idx]:
                        st.markdown(f'<div class="video-success">', unsafe_allow_html=True)
                        st.markdown(f"**Song {idx+1}**")
                        
                        # Watch button
                        st.markdown(
                            f'<a href="{video["url"]}" target="_blank">'
                            f'<button style="background-color: #4285F4; color: white; '
                            f'border: none; padding: 8px 16px; border-radius: 4px; '
                            f'cursor: pointer; width: 100%;">🔍 Search on YouTube</button></a>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Store session
            session_data = {
                "genre": genre_input,
                "artist": artist_name,
                "status": "NO_CHANNEL",
                "videos_provided": len(fallback_videos) if 'fallback_videos' in locals() else 0,
                "attempt": current_attempt,
                "timestamp": time.time()
            }
            st.session_state.sessions.append(session_data)

if __name__ == "__main__":
    main()
