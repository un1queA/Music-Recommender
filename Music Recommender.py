# prerequisites:
# pip install langchain
# pip install openai
# pip install langchain -U langchain-community
# pip install youtube-search
# pip install httpx==0.24.1
# pip install streamlit
# pip install streamlit -U streamlit
# pip install -U langchain-openai
# Run with this command ==> python -m streamlit run IAMUSIC2.py --logger.level=error

import streamlit as st
import os
import logging
import re
from youtube_search import YoutubeSearch
import time

# Set up logging
logging.getLogger("streamlit").setLevel(logging.WARNING)

# =============================================================================
# BACKEND FUNCTIONS - IMPROVED VERSION
# =============================================================================

def validate_api_key(api_key):
    """Check if the API key is a valid OpenAI GPT key"""
    if not api_key.startswith('sk-'):
        return False
    return True

def clean_artist_name(artist):
    """Clean artist name to avoid duplicates"""
    clean_name = re.sub(r'\([^)]*\)', '', artist).strip()
    return clean_name.lower()

def initialize_llm():
    """Initialize the LLM with the current API key"""
    try:
        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4", 
            temperature=1.0,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {e}")
        return None

def generate_artist_and_songs(genre):
    """Generate artist and song recommendations with duplicate prevention and bilingual titles"""
    
    llm = initialize_llm()
    if not llm:
        st.error("LLM not initialized. Please check your API key.")
        return None

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            # FIRST: Get artist with stage and real names
            artist_prompt = f"""I need ONE artist specializing in {genre}. 
            Provide in EXACT format:
            Stage Name: [artist stage name]
            Real Name: [real name or "Not widely published"]
            
            REQUIREMENTS:
            1. Artist must have official YouTube channel with music videos
            2. Avoid common names that cause confusion (like "Lisa" without context)
            3. If artist has real name, include it for accurate searching
            4. The artist should not be in this list: {list(st.session_state.used_artists)}
            
            Example for anime genre:
            Stage Name: LiSA
            Real Name: Risa Oribe"""
            
            artist_response = llm.predict(artist_prompt)
            
            # Parse artist info
            stage_name, real_name = None, None
            for line in artist_response.strip().split('\n'):
                line = line.strip()
                if line.startswith('Stage Name:'):
                    stage_name = line.replace('Stage Name:', '').strip()
                elif line.startswith('Real Name:'):
                    real_name = line.replace('Real Name:', '').strip()
            
            if not stage_name:
                continue
            
            clean_artist = clean_artist_name(stage_name)
            
            # Check for duplicates
            if clean_artist in st.session_state.used_artists:
                continue
            
            # SECOND: Get 3 songs with bilingual titles and avoid duplicates
            song_prompt = f"""Give me EXACTLY 3 popular songs from {stage_name} that have official music videos.
            Avoid these previously suggested songs: {list(st.session_state.used_songs.get(clean_artist, []))}
            
            Format EACH song as:
            Original: [original title] | Translation: [English translation/title]
            
            Example:
            1. Original: 紅蓮華 | Translation: Gurenge (Crimson Lotus)
            2. Original: 炎 | Translation: Homura (Flame)
            3. Original: ADAMAS | Translation: ADAMAS
            
            IMPORTANT:
            - Only 3 songs, no more no less
            - Each must have official music video
            - Use exact original titles
            - Include English translation when applicable
            - Avoid songs already suggested"""
            
            songs_response = llm.predict(song_prompt)
            
            # Parse songs
            song_list = []
            for line in songs_response.strip().split('\n'):
                line = line.strip()
                if 'Original:' in line and 'Translation:' in line:
                    # Remove numbering if present
                    if '.' in line:
                        line = line.split('.', 1)[1]
                    parts = line.split('|')
                    if len(parts) == 2:
                        original = parts[0].replace('Original:', '').strip()
                        translation = parts[1].replace('Translation:', '').strip()
                        song_list.append({
                            'original': original,
                            'translation': translation
                        })
            
            if len(song_list) == 3:
                # Update used artists and songs
                st.session_state.used_artists.add(clean_artist)
                if clean_artist not in st.session_state.used_songs:
                    st.session_state.used_songs[clean_artist] = []
                
                # Add songs to used list
                for song in song_list:
                    song_key = f"{song['original']}|{song['translation']}"
                    st.session_state.used_songs[clean_artist].append(song_key)
                
                return {
                    'stage_name': stage_name,
                    'real_name': real_name if real_name else "Not widely published",
                    'songs': song_list,
                    'clean_name': clean_artist
                }
                
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed, retrying...")
            continue
    
    st.error("Couldn't find a new artist after 5 attempts. Try a different genre or reset memory.")
    return None

def create_distinct_search_query(real_name, stage_name, genre):
    """Create distinct search query using real name as primary identifier"""
    if real_name and real_name != "Not widely published" and real_name.lower() != stage_name.lower():
        return f"{real_name} {stage_name} official {genre} singer"
    else:
        return f"{stage_name} official {genre} music"

def find_official_channel(artist_info, genre):
    """Find the official YouTube channel using real name for better accuracy"""
    distinct_query = create_distinct_search_query(artist_info['real_name'], artist_info['stage_name'], genre)
    
    try:
        channel_queries = [
            f'{distinct_query} channel official',
            f'{distinct_query} official youtube',
            f'{distinct_query} vevo',
            distinct_query,
        ]
        
        for query in channel_queries:
            try:
                results = YoutubeSearch(query, max_results=10).to_dict()
                
                for result in results:
                    channel_name = result['channel']
                    channel_lower = channel_name.lower()
                    stage_lower = artist_info['stage_name'].lower()
                    real_lower = artist_info['real_name'].lower() if artist_info['real_name'] != "Not widely published" else ""
                    
                    # Check for matches
                    stage_match = stage_lower in channel_lower
                    real_match = real_lower and real_lower in channel_lower
                    is_official = any(indicator in channel_lower for indicator in 
                                     ['official', 'vevo', 'topic'])
                    
                    if (real_match and is_official) or (stage_match and is_official):
                        return {
                            'channel_name': channel_name,
                            'channel_id': result.get('channel_id', ''),
                            'confidence': 0.8 if real_match else 0.6
                        }
                        
            except Exception:
                continue
        
        return None
        
    except Exception as e:
        return None

def search_songs_on_channel(artist_info, songs, channel_info):
    """Search for songs on specific channel"""
    if not channel_info:
        return []
    
    channel_name = channel_info['channel_name']
    found_videos = []
    
    for song in songs:
        try:
            # Use original title for search
            search_title = song['original']
            video_found = False
            
            # Search queries
            search_queries = [
                f'"{search_title}" {channel_name} official music video',
                f'"{search_title}" {artist_info["stage_name"]} {channel_name}',
                f'{search_title} {channel_name} mv',
            ]
            
            for query in search_queries:
                try:
                    results = YoutubeSearch(query, max_results=5).to_dict()
                    
                    for result in results:
                        if result['channel'].lower() != channel_name.lower():
                            continue
                        
                        video_title = result['title'].lower()
                        is_official = any(term in video_title for term in 
                                         ['official', 'music video', 'mv'])
                        not_wanted = any(term in video_title for term in 
                                        ['cover', 'tribute', 'fanmade', 'reaction'])
                        
                        if is_official and not not_wanted:
                            found_videos.append({
                                'song': song,
                                'video_id': result['id'],
                                'title': result['title'],
                                'type': 'official_mv',
                                'channel': channel_name
                            })
                            video_found = True
                            break
                    
                    if video_found:
                        break
                        
                except Exception:
                    continue
            
            # If no official video, look for alternatives
            if not video_found:
                alternate_types = ['live', 'lyric video', 'audio', 'performance']
                for alt_type in alternate_types:
                    try:
                        query = f'"{search_title}" {channel_name} {alt_type}'
                        results = YoutubeSearch(query, max_results=3).to_dict()
                        
                        for result in results:
                            if result['channel'].lower() == channel_name.lower():
                                video_title = result['title'].lower()
                                if 'cover' not in video_title:
                                    found_videos.append({
                                        'song': song,
                                        'video_id': result['id'],
                                        'title': result['title'],
                                        'type': alt_type,
                                        'channel': channel_name
                                    })
                                    video_found = True
                                    break
                        
                        if video_found:
                            break
                            
                    except Exception:
                        continue
                        
        except Exception:
            continue
    
    return found_videos

def calculate_relevance_score(video_title: str, channel_name: str, artist: str, song: str) -> float:
    """Calculate how relevant a YouTube video is to the requested artist and song"""
    score = 0.0
    
    clean_artist = re.sub(r'[^\w\s]', '', artist).strip()
    clean_song = re.sub(r'[^\w\s]', '', song).strip()
    clean_video_title = re.sub(r'[^\w\s]', '', video_title).strip()
    
    # 1. Check if artist is in channel name (very important)
    if clean_artist in channel_name:
        score += 0.4
    elif 'official' in channel_name and any(word in channel_name for word in clean_artist.split()):
        score += 0.3

    # 2. Check if song title appears in video title   
    if clean_song in clean_video_title:
        score += 0.3
    else:
        # Word-by-word matching as fallback
        song_words = set(clean_song.split())
        title_words = set(clean_video_title.split())
        
        matching_words = song_words.intersection(title_words)
        if len(song_words) > 0:
            title_match_ratio = len(matching_words) / len(song_words)
            score += title_match_ratio * 0.2
    
    # 3. Check for official indicators in title
    if 'official' in video_title and 'music video' in video_title:
        score += 0.2
    
    # 4. Penalize for unwanted content types
    unwanted_terms = ['lyric', 'lyrics', 'cover', 'tribute', 'fan', 'karaoke', 'instrumental', 'reaction']
    for term in unwanted_terms:
        if term in video_title:
            score -= 0.4
            break
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))

# =============================================================================
# FRONTEND FUNCTIONS
# =============================================================================

def setup_api_key():
    """Handle API key setup"""
    if st.session_state.get('api_key_valid') and st.session_state.get('hide_api_section', False):
        return
    
    st.sidebar.header("🔑 API Configuration")
    
    # Check if API key exists in environment
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and validate_api_key(env_key):
        os.environ["OPENAI_API_KEY"] = env_key
        st.session_state.api_key_valid = True
        st.sidebar.success("Using API key from environment")
        if 'hide_api_section' not in st.session_state:
            st.session_state.hide_api_section = True
        return
    
    # User input for API key
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    if st.sidebar.button("Validate Key"):
        if validate_api_key(api_key):
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.api_key_valid = True
            st.session_state.hide_api_section = True
            st.sidebar.success("✅ API Key validated!")
            time.sleep(2)
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid API key")

def reset_memory():
    """Reset memory to allow repeating artists"""
    st.session_state.used_artists = set()
    st.session_state.used_songs = {}
    st.success("Memory reset! You can now get the same artists again.")

def main_app():
    """Main application interface"""
    st.title("🎵 I AM MUSIC - Enhanced Version")
    
    # Reset button in sidebar
    if st.sidebar.button("🔄 Reset Memory", help="Clear used artists and songs memory"):
        reset_memory()
    
    genre = st.text_input("Enter a genre of music:", placeholder="e.g., anime, kpop, rock, jazz")

    if genre:
        with st.spinner("Generating recommendations..."):
            # Get artist and songs
            result = generate_artist_and_songs(genre)
            
            if result is None:
                return
            
            # Display artist info
            st.subheader(f"🎤 Recommended Artist")
            st.write(f"**Stage Name:** {result['stage_name']}")
            if result['real_name'] != "Not widely published":
                st.write(f"**Real Name:** {result['real_name']}")
            
            # Display songs with bilingual format
            st.subheader("🎵 Recommended Songs")
            for i, song in enumerate(result['songs'], 1):
                display_text = f"**{i}. {song['original']}**"
                if song['translation']:
                    display_text += f" ({song['translation']})"
                st.write(display_text)
            
            # Find official channel
            with st.spinner("🔍 Searching for official YouTube channel..."):
                channel_info = find_official_channel(result, genre)
                
                if channel_info:
                    st.success(f"✅ Found official channel: **{channel_info['channel_name']}**")
                    
                    # Search for songs on channel
                    with st.spinner(f"Searching for songs on {channel_info['channel_name']}..."):
                        found_videos = search_songs_on_channel(result, result['songs'], channel_info)
                    
                    # Display found videos
                    if found_videos:
                        st.subheader("🎬 Found Videos")
                        cols = st.columns(3)
                        
                        for idx, video in enumerate(found_videos[:3]):
                            with cols[idx % 3]:
                                # Display song info
                                song_display = video['song']['original']
                                if video['song']['translation']:
                                    song_display += f" ({video['song']['translation']})"
                                st.write(f"**{idx + 1}. {song_display}**")
                                
                                # Video type badge
                                type_badges = {
                                    'official_mv': '🎬 Official MV',
                                    'live': '🎤 Live',
                                    'lyric video': '📝 Lyric Video',
                                    'audio': '🔊 Audio',
                                    'performance': '🎭 Performance'
                                }
                                badge = type_badges.get(video['type'], video['type'])
                                st.caption(badge)
                                
                                # Display video
                                video_url = f"https://www.youtube.com/watch?v={video['video_id']}"
                                st.video(video_url)
                                
                                # Show video title
                                vid_title = video['title']
                                if len(vid_title) > 40:
                                    vid_title = vid_title[:40] + "..."
                                st.caption(f"*{vid_title}*")
                        
                        # Summary
                        official_count = sum(1 for v in found_videos if v['type'] == 'official_mv')
                        if official_count == 3:
                            st.success("🎉 Perfect! Found all 3 official music videos!")
                        elif official_count > 0:
                            st.success(f"✅ Found {official_count} official music videos and {len(found_videos) - official_count} alternate versions")
                        else:
                            st.info(f"ℹ️ Found {len(found_videos)} alternate versions (no official MVs available)")
                    else:
                        st.warning("⚠️ Could not find videos on the official channel. Trying general search...")
                        # Fallback to general search
                        for i, song in enumerate(result['songs'], 1):
                            st.write(f"**Searching for:** {song['original']}")
                            search_query = f"{result['stage_name']} {song['original']} official music video"
                            try:
                                results = YoutubeSearch(search_query, max_results=1).to_dict()
                                if results:
                                    video_url = f"https://www.youtube.com/watch?v={results[0]['id']}"
                                    st.video(video_url)
                            except:
                                st.caption("Video not found")
                else:
                    st.warning("⚠️ Could not find official channel. Showing general search results:")
                    # General fallback search
                    for i, song in enumerate(result['songs'], 1):
                        st.write(f"**{i}. {song['original']}** ({song['translation']})")
                        search_query = f"{result['stage_name']} {song['original']} official music video"
                        try:
                            results = YoutubeSearch(search_query, max_results=1).to_dict()
                            if results:
                                video_url = f"https://www.youtube.com/watch?v={results[0]['id']}"
                                st.video(video_url)
                                st.caption(f"Found: {results[0]['title']}")
                            else:
                                st.caption("❌ Video not found")
                        except Exception as e:
                            st.caption("❌ Search error")

# =============================================================================
# APP INITIALIZATION AND MAIN LOGIC
# =============================================================================

# Initialize session state
if 'used_artists' not in st.session_state:
    st.session_state.used_artists = set()

if 'used_songs' not in st.session_state:
    st.session_state.used_songs = {}

if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

if 'hide_api_section' not in st.session_state:
    st.session_state.hide_api_section = False

# Setup API key first
setup_api_key()

# Only run main app if API key is valid
if not st.session_state.api_key_valid:
    st.info("🔑 Please enter your OpenAI API key in the sidebar to use the app.")
else:
    main_app()
