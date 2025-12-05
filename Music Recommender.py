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
# INITIALIZATION
# =============================================================================

if 'used_artists' not in st.session_state:
    st.session_state.used_artists = set()

if 'used_songs' not in st.session_state:
    st.session_state.used_songs = {}

if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

if 'hide_api_section' not in st.session_state:
    st.session_state.hide_api_section = False

# =============================================================================
# BACKEND FUNCTIONS
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

def generate_search_terms(original_title, english_title, unique_phrase, language):
    """Generate multiple search strategies for a song based on its language"""
    search_sets = []
    
    # Strategy 1: Exact original title search (highest priority for non-English)
    if language != "English":
        search_sets.append({
            'name': 'Exact Original',
            'queries': [
                f'"{original_title}" official music video',
                f'"{original_title}" 公式MV' if language == "Japanese" else f'"{original_title}" MV',
                f'"{original_title}" 뮤직비디오' if language == "Korean" else f'"{original_title}" MV'
            ],
            'weight': 1.0
        })
    
    # Strategy 2: English title search (only if different and exists)
    if english_title and english_title.lower() != original_title.lower():
        search_sets.append({
            'name': 'English Title',
            'queries': [
                f'"{english_title}" official music video',
                f'"{english_title}" MV'
            ],
            'weight': 0.8
        })
    
    # Strategy 3: Unique phrase search (context-based)
    search_sets.append({
        'name': 'Unique Phrase',
        'queries': [
            f'"{unique_phrase}"',
            f'{unique_phrase} official'
        ],
        'weight': 0.9
    })
    
    # Strategy 4: Language-specific search terms
    if language != "English":
        # Add language-specific search terms
        lang_terms = {
            "Japanese": ["アニメ", "主題歌", "オープニング"],
            "Korean": ["K-pop", "가사", "뮤비"],
            "Chinese": ["中文歌曲", "MV", "官方"],
            "Spanish": ["español", "canción", "vídeo oficial"]
        }
        
        if language in lang_terms:
            for term in lang_terms[language]:
                search_sets.append({
                    'name': f'{language} Context',
                    'queries': [f'"{original_title}" {term}'],
                    'weight': 0.7
                })
    
    return search_sets

def detect_language(text):
    """Detect the primary language of a text"""
    # Simple language detection based on character ranges
    japanese_chars = any(0x3040 <= ord(char) <= 0x309F or  # Hiragana
                        0x30A0 <= ord(char) <= 0x30FF or  # Katakana
                        0x4E00 <= ord(char) <= 0x9FFF for char in text)  # Kanji
    
    korean_chars = any(0xAC00 <= ord(char) <= 0xD7A3 for char in text)
    chinese_chars = any(0x4E00 <= ord(char) <= 0x9FFF for char in text)
    
    if japanese_chars:
        return "Japanese"
    elif korean_chars:
        return "Korean"
    elif chinese_chars:
        return "Chinese"
    elif any(0x0400 <= ord(char) <= 0x04FF for char in text):  # Cyrillic
        return "Russian"
    elif any(0x0600 <= ord(char) <= 0x06FF for char in text):  # Arabic
        return "Arabic"
    else:
        # Check if it's mostly Latin characters
        latin_chars = sum(1 for char in text if 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isspace())
        if latin_chars / max(len(text), 1) > 0.7:
            return "English"
        else:
            return "Unknown"

def calculate_comprehensive_score(title, description, channel, artist_info, song_info, query):
    """Calculate score with multiple verification layers"""
    score = 0.0
    max_score = 1.0
    
    title_lower = title.lower()
    desc_lower = (description or '').lower()
    channel_lower = channel.lower()
    
    # 1. Artist verification (30% - MANDATORY)
    artist_indicators = [
        artist_info['stage_name'].lower(),
        artist_info['real_name'].lower() if artist_info['real_name'] != "Not widely published" else ""
    ]
    
    artist_found = False
    for indicator in artist_indicators:
        if indicator and indicator in title_lower:
            score += 0.15
            artist_found = True
        if indicator and indicator in channel_lower:
            score += 0.15
            artist_found = True
    
    # REJECTION RULE: If artist not found at all, reject completely
    if not artist_found:
        return 0.0
    
    # 2. Song title verification (40%)
    # Check original title (exact or partial)
    original_lower = song_info['original'].lower()
    if original_lower in title_lower:
        score += 0.4  # Exact match bonus
    else:
        # Check for partial matches with original title
        original_words = set(original_lower.split())
        title_words = set(title_lower.split())
        common_words = original_words.intersection(title_words)
        if common_words:
            score += (len(common_words) / max(len(original_words), 1)) * 0.3
    
    # 3. English title verification (bonus only)
    if song_info['english'] and song_info['english'].lower() in title_lower:
        score += 0.1  # Bonus for English title match
    
    # 4. Unique phrase verification (20%)
    if song_info['unique_phrase'].lower() in desc_lower:
        score += 0.2
    elif any(word.lower() in desc_lower for word in song_info['unique_phrase'].split()[:3]):
        score += 0.1
    
    # 5. Official indicators (10%)
    official_terms = ['official', 'vevo', 'music video', 'mv', '公式', '뮤직비디오', '공식']
    for term in official_terms:
        if term in title_lower or term in desc_lower:
            score += 0.03  # Smaller increments for multiple matches
            break
    
    # 6. Language consistency bonus
    detected_lang = detect_language(song_info['original'])
    title_lang = detect_language(title)
    if detected_lang == title_lang and detected_lang != "Unknown":
        score += 0.05
    
    # 7. Penalties for unwanted content
    unwanted = ['cover', 'tribute', 'reaction', 'lyrics', 'karaoke', 'instrumental', 'fan', 'reaccion', 'カバー']
    penalty_applied = False
    for term in unwanted:
        if term in title_lower:
            score -= 0.4  # Strong penalty
            penalty_applied = True
            break
    
    # 8. View count bonus (if available in description)
    if 'million' in desc_lower or '000,000' in desc_lower or 'views' in desc_lower:
        score += 0.05
    
    # 9. Channel name match bonus
    if artist_info.get('youtube_channel') and artist_info['youtube_channel'] != "Unknown":
        if artist_info['youtube_channel'].lower() in channel_lower:
            score += 0.1
    
    return min(max_score, max(0.0, score))

def find_and_verify_video(artist_info, song_info, search_strategies):
    """Find and VERIFY a video using multiple search strategies"""
    all_results = []
    
    for strategy in search_strategies:
        for query in strategy['queries']:
            try:
                # Add artist name to query for context
                enhanced_query = f"{query} {artist_info['stage_name']}"
                results = YoutubeSearch(enhanced_query, max_results=10).to_dict()
                
                for result in results:
                    score = calculate_comprehensive_score(
                        result['title'],
                        result.get('description', ''),
                        result['channel'],
                        artist_info,
                        song_info,
                        query
                    )
                    
                    if score >= 0.7:  # Good match threshold
                        result['strategy'] = strategy['name']
                        result['query'] = query
                        result['score'] = score
                        all_results.append(result)
                        
            except Exception:
                continue
    
    # Sort by score and remove duplicates
    if all_results:
        # Remove duplicates by video ID
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Return best result if score is high enough
        if unique_results and unique_results[0]['score'] >= 0.85:
            return unique_results[0]
    
    return None

def get_song_video_with_verification(artist_info, song):
    """Main function to get verified video for a song"""
    # Detect language for better search
    language = detect_language(song['original'])
    
    # Generate search strategies for this song
    search_strategies = generate_search_terms(
        song['original'], 
        song['english'], 
        song['unique_phrase'],
        language
    )
    
    # First attempt: Normal search with strategies
    video = find_and_verify_video(artist_info, song, search_strategies)
    
    if video:
        return {
            'video_id': video['id'],
            'title': video['title'],
            'channel': video['channel'],
            'strategy': video.get('strategy', 'Unknown'),
            'score': video.get('score', 0),
            'verified': video.get('score', 0) >= 0.85
        }
    
    # Second attempt: If artist has known YouTube channel
    if artist_info.get('youtube_channel') and artist_info['youtube_channel'] != "Unknown":
        try:
            channel_query = f"{song['original']} {artist_info['youtube_channel']}"
            results = YoutubeSearch(channel_query, max_results=5).to_dict()
            
            for result in results:
                if artist_info['youtube_channel'].lower() in result['channel'].lower():
                    score = calculate_comprehensive_score(
                        result['title'],
                        result.get('description', ''),
                        result['channel'],
                        artist_info,
                        song,
                        channel_query
                    )
                    
                    if score >= 0.7:
                        return {
                            'video_id': result['id'],
                            'title': result['title'],
                            'channel': result['channel'],
                            'strategy': 'Channel Search',
                            'score': score,
                            'verified': score >= 0.85
                        }
        except Exception:
            pass
    
    return None

def generate_artist_and_songs(genre):
    """Generate artist and songs with language-aware formatting"""
    llm = initialize_llm()
    if not llm:
        return None

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            # FIRST: Get artist with metadata
            artist_prompt = f"""I need ONE artist specializing in {genre}. 
            Provide in EXACT format:
            Stage Name: [artist stage name]
            Real Name: [real name or "Not widely published"]
            YouTube Channel Name: [exact name of their official YouTube channel if known]
            
            REQUIREMENTS:
            1. Artist MUST have official YouTube channel with music videos
            2. The YouTube channel name should be their verified/official channel
            3. Provide the most common/known channel name
            4. If unsure about channel, write "Unknown"
            
            Example for anime genre:
            Stage Name: LiSA
            Real Name: Risa Oribe
            YouTube Channel Name: LiSA Official Channel"""
            
            artist_response = llm.predict(artist_prompt)
            
            # Parse artist info
            stage_name, real_name, youtube_channel = None, None, None
            for line in artist_response.strip().split('\n'):
                line = line.strip()
                if line.startswith('Stage Name:'):
                    stage_name = line.replace('Stage Name:', '').strip()
                elif line.startswith('Real Name:'):
                    real_name = line.replace('Real Name:', '').strip()
                elif line.startswith('YouTube Channel Name:'):
                    youtube_channel = line.replace('YouTube Channel Name:', '').strip()
            
            if not stage_name:
                continue
            
            clean_artist = clean_artist_name(stage_name)
            
            # Check for duplicates
            if clean_artist in st.session_state.used_artists:
                continue
            
            # SECOND: Get 3 songs with language preservation
            song_prompt = f"""Give me EXACTLY 3 songs from {stage_name} with these requirements:
            
            CRITICAL REQUIREMENTS FOR EACH SONG:
            1. MUST have an official music video on YouTube
            2. MUST be from the artist's official channel
            3. Video should have at least 1 million views (popular)
            4. Song should be identifiable by a unique phrase in the video description
            
            FORMAT FOR EACH SONG (EXACTLY):
            Original: [exact title as it appears on YouTube in original language]
            English: [English translation or name if available, otherwise "None"]
            Unique Phrase: [5-7 word unique phrase from video description]
            Language: [Primary language of the song: Japanese, Korean, English, Chinese, Spanish, etc.]
            
            Example for LiSA (Japanese artist):
            Original: 紅蓮華
            English: Gurenge
            Unique Phrase: Demon Slayer Kimetsu no Yaiba opening theme song
            Language: Japanese
            
            Provide only 3 songs in this exact format, nothing else."""
            
            songs_response = llm.predict(song_prompt)
            
            # Parse songs with language info
            song_list = []
            current_song = {}
            
            for line in songs_response.strip().split('\n'):
                line = line.strip()
                if line.startswith('Original:'):
                    if current_song:  # Save previous song if exists
                        song_list.append(current_song)
                    current_song = {'original': line.replace('Original:', '').strip()}
                elif line.startswith('English:'):
                    current_song['english'] = line.replace('English:', '').strip()
                elif line.startswith('Unique Phrase:'):
                    current_song['unique_phrase'] = line.replace('Unique Phrase:', '').strip()
                elif line.startswith('Language:'):
                    current_song['language'] = line.replace('Language:', '').strip()
            
            # Add the last song
            if current_song:
                song_list.append(current_song)
            
            # Validate we got exactly 3 complete songs
            if len(song_list) == 3 and all('original' in s and 'english' in s and 'unique_phrase' in s and 'language' in s for s in song_list):
                # Update used artists
                st.session_state.used_artists.add(clean_artist)
                if clean_artist not in st.session_state.used_songs:
                    st.session_state.used_songs[clean_artist] = []
                
                return {
                    'stage_name': stage_name,
                    'real_name': real_name if real_name else "Not widely published",
                    'youtube_channel': youtube_channel if youtube_channel else "Unknown",
                    'songs': song_list,
                    'clean_name': clean_artist
                }
                
        except Exception as e:
            continue
    
    st.error("Couldn't find a suitable artist after 5 attempts.")
    return None

def format_song_display(song):
    """Format song display with original language as primary"""
    display = song['original']
    
    # Add English in parentheses only if:
    # 1. English exists and is not "None"
    # 2. Song is not in English
    # 3. English is different from original
    if (song.get('english') and 
        song['english'].lower() != 'none' and 
        song.get('language', '').lower() != 'english' and
        song['english'].lower() != song['original'].lower()):
        display += f" ({song['english']})"
    
    # Add language flag emoji
    language_flags = {
        'Japanese': '🇯🇵',
        'Korean': '🇰🇷', 
        'Chinese': '🇨🇳',
        'Spanish': '🇪🇸',
        'French': '🇫🇷',
        'German': '🇩🇪',
        'Italian': '🇮🇹',
        'Portuguese': '🇵🇹',
        'Russian': '🇷🇺',
        'Hindi': '🇮🇳',
        'Arabic': '🇸🇦',
        'Thai': '🇹🇭',
        'Vietnamese': '🇻🇳'
    }
    
    if song.get('language') in language_flags:
        display = f"{language_flags[song['language']]} {display}"
    
    return display

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

def main_app():
    """Main application interface with language-aware display"""
    st.title("🎵 I AM MUSIC - Multilingual Edition")
    
    if st.sidebar.button("🔄 Reset Memory"):
        st.session_state.used_artists = set()
        st.session_state.used_songs = {}
        st.success("Memory reset!")
    
    genre = st.text_input("Enter a music genre:", placeholder="e.g., anime, kpop, rock, jazz, latin pop")

    if genre:
        with st.spinner("🎵 Finding multilingual recommendations..."):
            # Get artist and songs
            result = generate_artist_and_songs(genre)
            
            if result is None:
                st.error("Failed to generate recommendations. Try a different genre.")
                return
            
            # Display artist info
            st.subheader(f"🎤 {result['stage_name']}")
            col1, col2 = st.columns(2)
            with col1:
                if result['real_name'] != "Not widely published":
                    st.caption(f"**Real name:** {result['real_name']}")
            with col2:
                if result['youtube_channel'] != "Unknown":
                    st.caption(f"**YouTube:** {result['youtube_channel']}")
            
            # Display songs
            st.subheader("🎵 Recommended Songs")
            
            for i, song in enumerate(result['songs'], 1):
                st.divider()
                
                # Song header with language info
                col_title, col_lang = st.columns([3, 1])
                with col_title:
                    display_name = format_song_display(song)
                    st.markdown(f"### {i}. {display_name}")
                with col_lang:
                    if song.get('language'):
                        st.caption(f"*{song['language']}*")
                
                # Unique phrase context
                if song.get('unique_phrase'):
                    st.info(f"🎯 **Context:** {song['unique_phrase']}")
                
                # Search and display video
                with st.spinner(f"🔍 Verifying video for {song['original']}..."):
                    video_info = get_song_video_with_verification(result, song)
                
                # Display result
                if video_info:
                    # Video found - show with confidence
                    confidence_color = "🟢" if video_info['score'] >= 0.9 else "🟡" if video_info['score'] >= 0.8 else "🟠"
                    st.success(f"{confidence_color} **Verified** (Confidence: {video_info['score']:.0%})")
                    
                    # Show video
                    video_url = f"https://www.youtube.com/watch?v={video_info['video_id']}"
                    st.video(video_url)
                    
                    # Verification details
                    with st.expander("📊 Verification Details"):
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Strategy", video_info['strategy'])
                        with cols[1]:
                            st.metric("Confidence", f"{video_info['score']:.0%}")
                        with cols[2]:
                            st.metric("Channel Match", "✅" if result['youtube_channel'].lower() in video_info['channel'].lower() else "❌")
                        
                        st.caption(f"**Channel:** {video_info['channel']}")
                        st.caption(f"**Video Title:** {video_info['title']}")
                        
                        # Language match indicator
                        detected_lang = detect_language(song['original'])
                        video_lang = detect_language(video_info['title'])
                        if detected_lang == video_lang and detected_lang != "Unknown":
                            st.success(f"✅ Language match: {detected_lang}")
                else:
                    # Video not found
                    st.error("❌ Could not verify official video")
                    
                    # Provide manual search options
                    with st.expander("🔍 Try searching manually"):
                        st.write("**Search these exact terms on YouTube:**")
                        
                        # Original language search
                        st.code(f"{result['stage_name']} {song['original']} official music video", language='bash')
                        
                        # English search if available
                        if song.get('english') and song['english'].lower() != 'none':
                            st.code(f"{result['stage_name']} {song['english']} official music video", language='bash')
                        
                        # Unique phrase search
                        if song.get('unique_phrase'):
                            st.code(song['unique_phrase'], language='bash')
            
            # Summary
            st.divider()
            found_count = 0
            for song in result['songs']:
                if get_song_video_with_verification(result, song):
                    found_count += 1
            
            if found_count == 3:
                st.balloons()
                st.success("🎉 Perfect! All 3 songs verified with high confidence!")
            elif found_count > 0:
                st.success(f"✅ Found {found_count} verified videos out of 3")
            else:
                st.info("💡 Use the manual search suggestions above to find the videos.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Setup API key first
setup_api_key()

# Only run main app if API key is valid
if not st.session_state.api_key_valid:
    st.info("🔑 Please enter your OpenAI API key in the sidebar to use the app.")
else:
    main_app()
