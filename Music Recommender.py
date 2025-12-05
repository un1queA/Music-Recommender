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
# BACKEND FUNCTIONS
# =============================================================================

def validate_api_key(api_key):
    """Check if the API key is a valid OpenAI GPT key"""
    if not api_key.startswith('sk-'):
        return False
    return True

def make_artist_distinct(artist, genre):
    """Make artist name distinct to avoid wrong searches on YouTube"""
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        prompt_template = PromptTemplate(
            input_variables=["artist", "genre"],
            template="""I need to make this artist name distinct for YouTube searching to avoid confusion with other artists.

            Original artist: {artist}
            Genre: {genre}

            Create a DISTINCT search query that will uniquely identify this specific artist on YouTube.
            
            IMPORTANT:
            1. Keep the original artist name
            2. Add unique identifiers that are specific to this artist
            3. Include genre-specific terms if helpful
            4. Add terms that would appear in their official channel name or about section
            5. The result should be a searchable string (max 10 words)
            6. Make it specific enough to filter out covers, tributes, and similar-named artists
            
            Examples:
            - "lisa" → "lisa blackpink official kpop"
            - "taylor swift" → "taylor swift official vevo pop singer"
            - "bts" → "bts bangtan boys official kpop group"
            - "周杰伦" → "周杰伦 jay chou official mandopop"
            
            Distinct search query for {artist} in {genre}:"""
        )
        
        chain = prompt_template | llm
        response = chain.invoke({"artist": artist, "genre": genre})
        
        distinct_query = response.content.strip()
        # Fallback if GPT fails
        if not distinct_query or len(distinct_query) < 3:
            return f"{artist} official {genre} music"
        
        return distinct_query
        
    except Exception as e:
        return f"{artist} official {genre} music"

def find_official_channel(distinct_artist_query):
    """Find the official YouTube channel using distinct artist query"""
    try:
        channel_queries = [
            f'{distinct_artist_query} channel official',
            f'{distinct_artist_query} official youtube',
            f'{distinct_artist_query} vevo',
            distinct_artist_query,
            f'"{distinct_artist_query.split()[0]}" official'  # Exact match for first word
        ]
        
        for query in channel_queries:
            try:
                results = YoutubeSearch(query, max_results=10).to_dict()
                
                for result in results:
                    channel_name = result['channel']
                    video_title = result['title'].lower()
                    channel_lower = channel_name.lower()
                    
                    # Strong indicators of official channel
                    is_official = any(indicator in channel_lower for indicator in 
                                     ['official', 'vevo', 'topic', 'music', 'records', 'label'])
                    
                    # Check for artist mentions
                    artist_keywords = distinct_artist_query.lower().split()
                    keyword_matches = sum(1 for keyword in artist_keywords if keyword in channel_lower)
                    
                    if (is_official and keyword_matches >= 2) or keyword_matches >= 3:
                        # Found potential official channel
                        return {
                            'channel_name': channel_name,
                            'channel_id': result.get('channel_id', ''),
                            'confidence': min(1.0, 0.3 + (keyword_matches * 0.15) + (0.2 if is_official else 0))
                        }
                        
            except Exception as e:
                continue
        
        # If no official channel found, try to find any channel with good match
        results = YoutubeSearch(distinct_artist_query, max_results=20).to_dict()
        best_channel = None
        best_score = 0
        
        for result in results:
            channel_name = result['channel']
            channel_lower = channel_name.lower()
            
            artist_keywords = distinct_artist_query.lower().split()
            keyword_matches = sum(1 for keyword in artist_keywords if keyword in channel_lower)
            
            score = keyword_matches / max(1, len(artist_keywords))
            
            if score > best_score:
                best_score = score
                best_channel = {
                    'channel_name': channel_name,
                    'channel_id': result.get('channel_id', ''),
                    'confidence': score
                }
        
        return best_channel
        
    except Exception as e:
        return None

def search_songs_on_channel(artist, songs, channel_info):
    """Search for songs on a specific channel and find official music videos"""
    if not channel_info:
        return []
    
    channel_name = channel_info['channel_name']
    found_videos = []
    
    st.info(f"🔍 Searching for songs on channel: {channel_name}")
    
    for song in songs[:3]:  # Only search for first 3 songs
        try:
            # Try to find official music video first
            video_found = False
            
            # Strategy 1: Search within channel for official music video
            search_queries = [
                f'"{song}" site:youtube.com/c/{channel_name.replace(" ", "")} official music video',
                f'{song} {artist} {channel_name} official',
                f'{song} {channel_name} music video',
                f'{song} {artist} mv',
            ]
            
            for query in search_queries:
                try:
                    results = YoutubeSearch(query, max_results=5).to_dict()
                    
                    for result in results:
                        video_title = result['title'].lower()
                        result_channel = result['channel']
                        
                        # Check if from correct channel
                        if result_channel.lower() != channel_name.lower():
                            continue
                        
                        # Check for official indicators
                        is_official = any(term in video_title for term in 
                                         ['official', 'music video', 'mv', 'vevo'])
                        
                        # Check for unwanted content
                        not_wanted = any(term in video_title for term in 
                                        ['cover', 'lyric', 'karaoke', 'reaction', 'review', 'tribute'])
                        
                        if is_official and not not_wanted:
                            video_info = {
                                'song': song,
                                'video_id': result['id'],
                                'title': result['title'],
                                'type': 'official_mv',
                                'channel': result_channel
                            }
                            found_videos.append(video_info)
                            video_found = True
                            break
                            
                    if video_found:
                        break
                        
                except Exception as e:
                    continue
            
            # Strategy 2: If no official MV found, look for alternate versions
            if not video_found:
                alternate_types = ['live performance', 'lyric video', 'audio', 'performance']
                
                for alt_type in alternate_types:
                    try:
                        query = f'{song} {channel_name} {alt_type}'
                        results = YoutubeSearch(query, max_results=3).to_dict()
                        
                        for result in results:
                            if result['channel'].lower() == channel_name.lower():
                                video_title = result['title'].lower()
                                
                                # Make sure it's not a cover
                                if 'cover' not in video_title and 'tribute' not in video_title:
                                    video_info = {
                                        'song': song,
                                        'video_id': result['id'],
                                        'title': result['title'],
                                        'type': alt_type,
                                        'channel': channel_name
                                    }
                                    found_videos.append(video_info)
                                    video_found = True
                                    break
                                    
                        if video_found:
                            break
                            
                    except Exception as e:
                        continue
            
            # Strategy 3: Last resort - any video from this channel with the song title
            if not video_found:
                try:
                    query = f'{song} {channel_name}'
                    results = YoutubeSearch(query, max_results=3).to_dict()
                    
                    for result in results:
                        if result['channel'].lower() == channel_name.lower():
                            video_info = {
                                'song': song,
                                'video_id': result['id'],
                                'title': result['title'],
                                'type': 'channel_video',
                                'channel': channel_name
                            }
                            found_videos.append(video_info)
                            break
                            
                except Exception as e:
                    pass
                    
        except Exception as e:
            continue
    
    return found_videos

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
    """Generate artist and EXACTLY 3 songs that likely have official music videos"""
    
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    
    llm = initialize_llm()
    if not llm:
        st.error("LLM not initialized. Please check your API key.")
        return None

    # Prompt to get artist with distinct name considerations
    prompt_template_genre = PromptTemplate(
        input_variables=["genre"],
        template="""I am trying to discover new music. Give me ONE artist or singer that specializes in {genre}. 
        
        IMPORTANT REQUIREMENTS:
        1. Suggest ONLY ONE artist
        2. The artist MUST have a distinct name that won't be confused with others on YouTube
        3. The artist should have an active official YouTube channel
        4. The artist should have multiple official music videos
        5. Use their official/recognizable name (native language preferred for non-English artists)
        6. Avoid artists with very common names (like "John", "Smith", "Lisa" without context)
        
        Good examples:
        - "BLACKPINK" (not just "Lisa" or "Jennie")
        - "Taylor Swift" (distinct full name)
        - "BTS" (unique group name)
        - "宇多田ヒカル" (Utada Hikaru in Japanese)
        - "The Weeknd" (unique spelling)
        
        Artist specializing in {genre}:"""
    )
    
    # Prompt to get exactly 3 songs with official music videos
    prompt_template_song = PromptTemplate(
        input_variables=["artist_suggestion"],
        template="""Give me EXACTLY 3 songs from {artist_suggestion} that DEFINITELY have official music videos on YouTube.
        
        CRITICAL REQUIREMENTS:
        1. Return EXACTLY 3 songs, no more no less
        2. Each song MUST have an official music video
        3. Use the original song titles exactly as they appear on YouTube
        4. These should be popular songs with high view counts
        5. Format: One song per line, no numbering or bullet points
        6. Only include songs where the official video is on the artist's official channel
        
        Example format:
        DDU-DU DDU-DU
        How You Like That
        Kill This Love
        
        3 songs from {artist_suggestion} with official music videos:"""
    )

    # Create chains
    genre_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_genre,
        output_key="artist_suggestion"
    )

    song_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_song,
        output_key="song_suggestion"
    )

    # Create sequential chain
    chain = SequentialChain(
        chains=[genre_chain, song_chain],
        input_variables=["genre"],
        output_variables=["artist_suggestion", "song_suggestion"],
        verbose=False
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        response = chain({"genre": genre})
        artist = response["artist_suggestion"].strip()
        
        # Parse songs
        raw_songs = [s.strip() for s in response["song_suggestion"].split('\n') if s.strip()]
        songs = raw_songs[:3]  # Take only first 3
        
        if len(songs) == 3:
            return {
                "artist": artist,
                "songs": songs
            }
    
    return None

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

def display_video_results(artist, songs, found_videos):
    """Display the found videos in a nice format"""
    st.subheader(f"🎵 {artist} - Music Videos Found")
    
    # Create columns for layout
    cols = st.columns(3)
    
    for idx, video_info in enumerate(found_videos[:3]):  # Show max 3 videos
        with cols[idx % 3]:
            song = video_info['song']
            video_id = video_info['video_id']
            video_type = video_info['type']
            video_title = video_info['title']
            
            # Display song name
            st.write(f"**{idx+1}. {song}**")
            
            # Display video type badge
            type_colors = {
                'official_mv': '🎬 Official MV',
                'live performance': '🎤 Live',
                'lyric video': '📝 Lyric Video',
                'audio': '🔊 Audio',
                'performance': '🎭 Performance',
                'channel_video': '📺 Channel Video'
            }
            
            badge = type_colors.get(video_type, video_type)
            st.caption(f"{badge}")
            
            # Display video
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            st.video(video_url)
            
            # Show video title (truncated if too long)
            if len(video_title) > 40:
                video_title = video_title[:40] + "..."
            st.caption(f"*{video_title}*")
    
    # Show summary
    official_count = sum(1 for v in found_videos if v['type'] == 'official_mv')
    
    if official_count == 3:
        st.success("🎉 Perfect! Found all 3 official music videos!")
    elif official_count > 0:
        st.success(f"✅ Found {official_count} official music videos and {3 - official_count} alternate versions")
    elif found_videos:
        st.info(f"ℹ️ Found {len(found_videos)} alternate versions (no official MVs available)")
    else:
        st.warning("⚠️ No videos found. The artist may not have official YouTube content.")

def main_app():
    """Main application interface following the agenda"""
    st.title("🎵 I AM MUSIC - Official Channel Finder")
    
    genre = st.text_input("Enter a music genre:", placeholder="e.g., kpop, rock, hip hop, jazz")
    
    if genre and st.button("Find Artist & Videos", type="primary"):
        with st.spinner("🔍 Finding distinct artist and songs..."):
            # Step 1: Generate artist and exactly 3 songs
            result = generate_artist_and_songs(genre)
            
            if not result:
                st.error("Could not generate artist and songs. Please try a different genre.")
                return
            
            artist = result["artist"]
            songs = result["songs"]
            
            # Display artist info
            st.subheader(f"🎤 Recommended Artist: {artist}")
            
            # Display songs
            st.write("**Selected Songs:**")
            for i, song in enumerate(songs, 1):
                st.write(f"{i}. {song}")
            
            # Step 2: Make artist name distinct for YouTube search
            with st.spinner("Making artist name distinct..."):
                distinct_query = make_artist_distinct(artist, genre)
                st.info(f"🔍 Distinct search: *{distinct_query}*")
            
            # Step 3: Find official YouTube channel
            with st.spinner("Searching for official YouTube channel..."):
                channel_info = find_official_channel(distinct_query)
                
                if channel_info:
                    st.success(f"✅ Found channel: **{channel_info['channel_name']}**")
                else:
                    st.warning("⚠️ Could not find official channel. Trying alternative search...")
                    # Fallback to simpler search
                    simple_query = f"{artist} official"
                    channel_info = find_official_channel(simple_query)
            
            # Step 4: Search for songs on the channel
            if channel_info:
                with st.spinner(f"Searching for songs on {channel_info['channel_name']}..."):
                    found_videos = search_songs_on_channel(artist, songs, channel_info)
                
                # Step 5: Display results
                display_video_results(artist, songs, found_videos)
                
                # Show channel info
                with st.expander("📺 Channel Details"):
                    st.write(f"**Channel Name:** {channel_info['channel_name']}")
                    st.write(f"**Confidence:** {channel_info['confidence']:.0%}")
                    
                    if channel_info.get('channel_id'):
                        channel_url = f"https://www.youtube.com/channel/{channel_info['channel_id']}"
                        st.write(f"**Channel URL:** [Click here]({channel_url})")
            else:
                st.error("❌ Could not find artist's YouTube channel. Please try a different genre or artist.")

# =============================================================================
# APP INITIALIZATION AND MAIN LOGIC
# =============================================================================

# Initialize session state
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
