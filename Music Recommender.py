# AI Music Recommender - app.py
# Run: streamlit run app.py

import streamlit as st
import re
import time
from typing import Dict, List, Optional, Tuple
from youtube_search import YoutubeSearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_RETRY_ATTEMPTS = 5  # Try up to 5 different artists
SONGS_REQUIRED = 3      # Must find exactly 3 songs

# =============================================================================
# LLM INITIALIZATION
# =============================================================================

def initialize_llm(api_key: str) -> Optional[ChatOpenAI]:
    """Initialize DeepSeek LLM."""
    try:
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0.4,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            max_tokens=1500,
            timeout=30
        )
    except Exception as e:
        st.error(f"Failed to initialize DeepSeek: {e}")
        return None

# =============================================================================
# ARTIST DISCOVERY
# =============================================================================

def find_artist_for_genre(genre: str, excluded_artists: List[str], llm: ChatOpenAI) -> Optional[str]:
    """Find a new artist for the given genre."""
    
    excluded_text = ", ".join(excluded_artists[-8:]) if excluded_artists else "None"
    
    prompt = PromptTemplate(
        input_variables=["genre", "excluded"],
        template="""Find ONE popular artist from the "{genre}" genre.

Already suggested (DO NOT repeat): {excluded}

Requirements:
- Must be a well-known artist with official YouTube channel
- Must have released music before 2024
- Must be different from excluded artists

Return ONLY the artist name, nothing else.
Example: "Taylor Swift" or "BTS" or "AR Rahman"

ARTIST:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"genre": genre, "excluded": excluded_text})
        artist = result.strip().strip('"').strip("'")
        
        # Clean up common artifacts
        if "ARTIST:" in artist:
            artist = artist.split("ARTIST:")[1].strip()
        
        return artist if artist else None
    except:
        return None

# =============================================================================
# CHANNEL DISCOVERY & VALIDATION
# =============================================================================

def find_official_channel(artist_name: str, llm: ChatOpenAI) -> Optional[str]:
    """Find the artist's official YouTube channel with validation."""
    
    # Generate smart search queries
    search_queries = [
        f"{artist_name} VEVO",           # VEVO channels are reliable
        f"{artist_name} official",       # Official channels
        f'"{artist_name}"',              # Exact name
        artist_name                      # Simple search
    ]
    
    channel_candidates = {}
    
    # Search and gather candidates
    for query in search_queries[:3]:  # Use top 3 queries
        try:
            results = YoutubeSearch(query, max_results=20).to_dict()
            
            for result in results:
                channel = result['channel'].strip()
                channel_lower = channel.lower()
                
                # Skip obvious non-official channels
                if 'topic' in channel_lower or 'lyrics' in channel_lower:
                    continue
                
                if channel not in channel_candidates:
                    channel_candidates[channel] = {
                        'videos': [],
                        'score': 0
                    }
                
                channel_candidates[channel]['videos'].append(result['title'])
                
                # Score the channel
                score = calculate_channel_score(channel, artist_name)
                channel_candidates[channel]['score'] = max(
                    channel_candidates[channel]['score'], 
                    score
                )
            
            time.sleep(0.2)
        except:
            continue
    
    if not channel_candidates:
        return None
    
    # Sort by score
    sorted_channels = sorted(
        channel_candidates.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    # Validate top candidates with LLM
    for channel_name, data in sorted_channels[:3]:
        if data['score'] < 80:
            continue
        
        # Ask LLM if this is the artist's official channel
        if validate_channel_with_llm(channel_name, artist_name, data['videos'][:5], llm):
            return channel_name
    
    # Fallback: return highest scored
    return sorted_channels[0][0] if sorted_channels else None

def calculate_channel_score(channel_name: str, artist_name: str) -> int:
    """Score how likely this channel is the artist's official channel."""
    score = 0
    
    channel_clean = re.sub(r'[^\w\s]', '', channel_name.lower())
    artist_clean = re.sub(r'[^\w\s]', '', artist_name.lower())
    
    # Exact match
    if channel_clean == artist_clean:
        score += 200
    # Contains artist name
    elif artist_clean in channel_clean:
        # Check if channel name is not too long (probably company)
        if len(channel_clean.split()) <= len(artist_clean.split()) + 2:
            score += 120
        else:
            score += 40
    
    # VEVO is always official
    if 'vevo' in channel_clean:
        score += 150
    
    # Official keyword
    if 'official' in channel_clean:
        score += 50
    
    return score

def validate_channel_with_llm(channel_name: str, artist_name: str, sample_videos: List[str], llm: ChatOpenAI) -> bool:
    """Use LLM to validate if channel belongs to the artist."""
    
    videos_text = "\n".join([f"- {v}" for v in sample_videos])
    
    prompt = PromptTemplate(
        input_variables=["channel", "artist", "videos"],
        template="""Determine if this YouTube channel belongs to the artist.

Channel: "{channel}"
Artist: "{artist}"

Sample videos:
{videos}

Question: Is "{channel}" the OFFICIAL channel of {artist}?

Consider:
- Does the channel name match the artist?
- Are the videos BY this artist (not a record label with multiple artists)?
- VEVO channels are official artist channels

Answer ONLY: YES or NO

ANSWER:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({
            "channel": channel_name,
            "artist": artist_name,
            "videos": videos_text
        })
        return "YES" in result.upper()
    except:
        return False

# =============================================================================
# VIDEO DISCOVERY & FILTERING
# =============================================================================

def discover_videos_from_channel(channel_name: str, artist_name: str, llm: ChatOpenAI) -> List[Dict]:
    """Find music videos from the locked channel."""
    
    all_videos = []
    
    # Multiple search strategies
    searches = [
        channel_name,
        f"{channel_name} music video",
        f"{artist_name} {channel_name}",
        f"{channel_name} official"
    ]
    
    for search in searches[:3]:
        try:
            results = YoutubeSearch(search, max_results=50).to_dict()
            
            for result in results:
                # STRICT: Must be from exact channel
                if not channels_match(result['channel'].strip(), channel_name):
                    continue
                
                # Basic filtering
                title_lower = result['title'].lower()
                if any(word in title_lower for word in [
                    'behind the scenes', 'interview', 'reaction', 
                    'dance practice', 'choreography'
                ]):
                    continue
                
                # Score the video
                score = score_music_video(result)
                
                if score >= 30:
                    all_videos.append({
                        'id': result['id'],
                        'url': f"https://www.youtube.com/watch?v={result['id']}",
                        'title': result['title'],
                        'duration': result.get('duration', ''),
                        'views': result.get('views', ''),
                        'score': score
                    })
        except:
            continue
    
    # Remove duplicates
    unique_videos = {}
    for video in all_videos:
        if video['id'] not in unique_videos:
            unique_videos[video['id']] = video
    
    videos_list = list(unique_videos.values())
    
    # Sort by score
    videos_list.sort(key=lambda x: x['score'], reverse=True)
    
    return videos_list

def channels_match(channel1: str, channel2: str) -> bool:
    """Check if two channel names match (fuzzy matching)."""
    c1 = re.sub(r'[^\w]', '', channel1.lower())
    c2 = re.sub(r'[^\w]', '', channel2.lower())
    return c1 == c2

def score_music_video(video: Dict) -> int:
    """Score video likelihood of being a music video."""
    score = 0
    title_lower = video['title'].lower()
    
    # Duration (2-10 minutes ideal)
    if video.get('duration'):
        try:
            parts = video['duration'].split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 2 <= mins <= 10:
                    score += 60
                elif 1 <= mins <= 15:
                    score += 30
        except:
            pass
    
    # Title keywords
    if any(word in title_lower for word in ['official', 'music video', 'mv', 'audio', 'lyric']):
        score += 40
    
    # Views
    if video.get('views'):
        views = video['views'].lower()
        if 'm' in views:
            score += 30
        elif 'k' in views:
            score += 15
    
    return score

# =============================================================================
# SONG SELECTION & UNIQUENESS
# =============================================================================

def select_unique_songs(videos: List[Dict], count: int, llm: ChatOpenAI) -> List[Dict]:
    """Select unique songs using LLM to prevent duplicates."""
    
    if len(videos) < count:
        return videos
    
    selected = []
    
    for video in videos:
        if len(selected) >= count:
            break
        
        # Check if this is a unique song
        if not selected:
            selected.append(video)
            continue
        
        is_unique = True
        for selected_video in selected:
            if not are_different_songs(video['title'], selected_video['title'], llm):
                is_unique = False
                break
        
        if is_unique:
            selected.append(video)
    
    return selected

def are_different_songs(title1: str, title2: str, llm: ChatOpenAI) -> bool:
    """Check if two video titles represent different songs."""
    
    # Quick check: if base titles are very similar, they're the same
    base1 = title1.lower().split('(')[0].split('[')[0].strip()
    base2 = title2.lower().split('(')[0].split('[')[0].strip()
    
    if base1 == base2:
        return False
    
    # Use LLM for nuanced check
    prompt = PromptTemplate(
        input_variables=["title1", "title2"],
        template="""Are these two videos DIFFERENT songs?

Video 1: "{title1}"
Video 2: "{title2}"

Same song examples:
- "Dynamite" vs "Dynamite (Official Video)"
- "Shape of You" vs "Shape of You [Official Audio]"

Answer ONLY: YES or NO

ANSWER:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run({"title1": title1, "title2": title2})
        return "YES" in result.upper()
    except:
        return base1 != base2

# =============================================================================
# MAIN DISCOVERY PIPELINE
# =============================================================================

def discover_songs(genre: str, excluded_artists: List[str], llm: ChatOpenAI) -> Optional[Dict]:
    """
    Main pipeline: Find artist → Find channel → Find 3 unique songs
    Returns None if unsuccessful.
    """
    
    # Step 1: Find artist
    artist = find_artist_for_genre(genre, excluded_artists, llm)
    if not artist:
        return None
    
    # Step 2: Find official channel
    channel = find_official_channel(artist, llm)
    if not channel:
        return None
    
    # Step 3: Discover videos
    videos = discover_videos_from_channel(channel, artist, llm)
    if len(videos) < SONGS_REQUIRED:
        return None
    
    # Step 4: Select unique songs
    selected = select_unique_songs(videos, SONGS_REQUIRED, llm)
    if len(selected) < SONGS_REQUIRED:
        return None
    
    return {
        'artist': artist,
        'channel': channel,
        'songs': selected
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Music Recommender",
        page_icon="🎵",
        layout="wide"
    )
    
    # Session state
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'excluded_by_genre' not in st.session_state:
        st.session_state.excluded_by_genre = {}
    if 'total_searches' not in st.session_state:
        st.session_state.total_searches = 0
    
    # Custom CSS
    st.markdown("""
    <style>
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    .video-card {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎵 AI Music Recommender")
    st.caption("Discover music from any genre, any language")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-..."
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.write(f"Total searches: {st.session_state.total_searches}")
        
        if st.session_state.excluded_by_genre:
            st.write("**Artists discovered by genre:**")
            for genre, artists in list(st.session_state.excluded_by_genre.items())[:5]:
                st.write(f"• {genre}: {len(artists)} artists")
        
        st.markdown("---")
        if st.button("🔄 Reset Session"):
            st.session_state.excluded_by_genre = {}
            st.session_state.total_searches = 0
            st.rerun()
    
    # Main content
    st.markdown("### Enter a Music Genre")
    
    with st.form("genre_form"):
        genre = st.text_input(
            "Genre",
            placeholder="e.g., K-pop, J-pop, Bollywood, Reggaeton, Jazz...",
            label_visibility="collapsed"
        )
        
        submit = st.form_submit_button("🔍 Discover Music", use_container_width=True)
    
    # Process search
    if submit and genre:
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your DeepSeek API key in the sidebar!")
            return
        
        llm = initialize_llm(st.session_state.api_key)
        if not llm:
            return
        
        # Get excluded artists for this genre
        genre_key = genre.lower().strip()
        if genre_key not in st.session_state.excluded_by_genre:
            st.session_state.excluded_by_genre[genre_key] = []
        
        excluded = st.session_state.excluded_by_genre[genre_key]
        st.session_state.total_searches += 1
        
        # Retry loop
        progress_placeholder = st.empty()
        result = None
        
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            progress_placeholder.info(
                f"🔄 Attempt {attempt}/{MAX_RETRY_ATTEMPTS}: "
                f"Searching for artist with 3 songs..."
            )
            
            result = discover_songs(genre, excluded, llm)
            
            if result:
                # Success!
                artist = result['artist']
                
                # Check if already suggested
                if artist in excluded:
                    continue
                
                # Add to excluded
                excluded.append(artist)
                
                # Display results
                progress_placeholder.empty()
                
                st.markdown(
                    f'<div class="success-box">'
                    f'🎤 Artist: {artist}<br>'
                    f'📺 Channel: {result["channel"]}<br>'
                    f'🎵 Found {len(result["songs"])} unique songs'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                st.markdown(f"### 🎵 {artist} - Top Songs")
                
                cols = st.columns(3)
                
                for idx, song in enumerate(result['songs']):
                    with cols[idx]:
                        st.markdown(f"**Song {idx + 1}**")
                        
                        # Embed video
                        try:
                            st.video(song['url'])
                        except:
                            video_id = song['id']
                            st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                        
                        # Title
                        title = song['title']
                        if len(title) > 50:
                            title = title[:47] + "..."
                        st.write(f"**{title}**")
                        
                        # Metadata
                        if song.get('duration'):
                            st.caption(f"⏱️ {song['duration']}")
                        if song.get('views'):
                            st.caption(f"👁️ {song['views']}")
                        
                        # Watch button
                        st.markdown(
                            f'<a href="{song["url"]}" target="_blank">'
                            '<button style="background:#FF0000;color:white;'
                            'border:none;padding:10px 20px;border-radius:5px;'
                            'cursor:pointer;width:100%">'
                            '▶️ Watch on YouTube</button></a>',
                            unsafe_allow_html=True
                        )
                
                # Show remaining
                remaining = 50 - len(excluded)
                if remaining > 0:
                    st.success(
                        f"✅ Search again for more {genre} artists! "
                        f"(~{remaining} more available)"
                    )
                else:
                    st.info("You've explored many artists in this genre!")
                
                break
            
            # Failed, try next artist
            if result and result.get('artist'):
                excluded.append(result['artist'])
        
        # All attempts failed
        if not result or attempt == MAX_RETRY_ATTEMPTS:
            progress_placeholder.error(
                f"❌ Could not find a suitable artist with 3 songs after "
                f"{MAX_RETRY_ATTEMPTS} attempts. This genre might be too niche "
                f"or have limited YouTube content."
            )

if __name__ == "__main__":
    main()
