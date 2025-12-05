import streamlit as st
import os
import logging
import re
from youtube_search import YoutubeSearch
import time

# Set up logging
logging.getLogger("streamlit").setLevel(logging.WARNING)

# =============================================================================
# BACKEND FUNCTIONS - UPDATED
# =============================================================================

def validate_api_key(api_key):
    """Check if the API key is a valid OpenAI GPT key"""
    if not api_key.startswith('sk-'):
        return False
    return True

# --- NEW FUNCTION: Dynamically creates a unique search string using real name ---
def create_distinct_search_query(real_name, stage_name, genre):
    """Creates a distinct YouTube search query using real name as the primary key."""
    # Use the real name as the primary identifier. If not available, fall back to the stage name.
    primary_name = real_name if real_name.lower() != "not widely published" else stage_name
    distinct_query = f"{primary_name} {stage_name} official {genre} singer"
    # Fallback logic if real name is the same as stage name or not helpful
    if primary_name == stage_name:
        distinct_query = f"{stage_name} official {genre} artist channel"
    return distinct_query

def make_artist_distinct(artist_info, genre):
    """
    Creates a distinct search query. Now uses structured artist_info dict.
    artist_info dict must contain: 'stage_name' and 'real_name'.
    """
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        prompt_template = PromptTemplate(
            input_variables=["stage_name", "real_name", "genre"],
            template="""I need to make a search query for YouTube to find the official channel of a specific artist and avoid confusion.

            Stage Name: {stage_name}
            Real Name: {real_name}
            Genre: {genre}

            Create a DISTINCT YouTube search query (max 10 words) that uses the real name as the PRIMARY identifier to find THIS specific artist's official content.

            Examples:
            - For LiSA (Risa Oribe) in anime: "Risa Oribe LiSA official anime singer channel"
            - For BTS (Bangtan Sonyeondan) in K-pop: "Bangtan Sonyeondan BTS official kpop group channel"
            - For Taylor Swift in pop: "Taylor Swift official pop singer vevo"

            Distinct search query:"""
        )

        chain = prompt_template | llm
        response = chain.invoke({"stage_name": artist_info['stage_name'],
                                 "real_name": artist_info['real_name'],
                                 "genre": genre})

        distinct_query = response.content.strip()
        if not distinct_query or len(distinct_query) < 3:
            # Fallback to the programmatic function if LLM fails
            return create_distinct_search_query(artist_info['real_name'], artist_info['stage_name'], genre)
        return distinct_query

    except Exception as e:
        # Fallback to the programmatic function on any error
        return create_distinct_search_query(artist_info['real_name'], artist_info['stage_name'], genre)

# --- UPDATED FUNCTION: Now uses the new distinct search logic ---
def find_official_channel(artist_info, genre):
    """
    Find the official YouTube channel.
    artist_info dict must contain: 'stage_name' and 'real_name'.
    """
    distinct_query = make_artist_distinct(artist_info, genre)
    # st.info(f"🔍 Distinct search: *{distinct_query}*") # Optional: display the query

    try:
        channel_queries = [
            f'{distinct_query} channel',
            f'{distinct_query}',
            f'"{artist_info["real_name"]}" "{artist_info["stage_name"]}" official',  # Exact phrase
            f'"{artist_info["stage_name"]}" official {genre}'  # Fallback
        ]

        for query in channel_queries:
            try:
                results = YoutubeSearch(query, max_results=10).to_dict()
                for result in results:
                    channel_name = result['channel']
                    channel_lower = channel_name.lower()
                    # Keywords from our query that should ideally be in the channel name
                    search_keywords = distinct_query.lower().split()
                    keyword_matches = sum(1 for keyword in search_keywords if keyword in channel_lower)
                    # Strong official indicators
                    is_official = any(indicator in channel_lower for indicator in
                                     ['official', 'vevo', 'topic'])
                    if (is_official and keyword_matches >= 2) or keyword_matches >= 3:
                        return {
                            'channel_name': channel_name,
                            'channel_id': result.get('channel_id', ''),
                            'confidence': min(1.0, 0.3 + (keyword_matches * 0.15) + (0.2 if is_official else 0))
                        }
            except Exception as e:
                continue
        return None
    except Exception as e:
        return None

# --- UPDATED FUNCTION: Now receives song dicts ---
def search_songs_on_channel(artist_info, song_list, channel_info):
    """
    Search for songs on a specific channel.
    artist_info dict: contains 'stage_name', 'real_name'.
    song_list: list of dicts with 'original_title' and 'english_title'.
    channel_info dict: contains 'channel_name'.
    """
    if not channel_info:
        return []
    channel_name = channel_info['channel_name']
    found_videos = []
    stage_name = artist_info['stage_name']

    # st.info(f"🔍 Searching on channel: {channel_name}") # Optional logging

    for song_dict in song_list[:3]:
        original_title = song_dict['original_title']
        # Use original title for search, as it's most likely in the video title.
        search_song_title = original_title
        try:
            video_found = False
            # STRATEGY 1: Search for official music video on the channel
            search_queries = [
                f'"{search_song_title}" {channel_name} official music video',
                f'"{search_song_title}" {stage_name} {channel_name}',
                f'{search_song_title} {channel_name} mv',
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
                                        ['cover', 'tribute', 'fanmade'])
                        if is_official and not not_wanted:
                            found_videos.append({
                                'song_dict': song_dict, # Store the full song info
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

            # STRATEGY 2: If no official MV, look for live/lyric/audio from the channel
            if not video_found:
                alternate_types = ['live', 'lyric video', 'audio', 'performance']
                for alt_type in alternate_types:
                    try:
                        query = f'"{search_song_title}" {channel_name} {alt_type}'
                        results = YoutubeSearch(query, max_results=3).to_dict()
                        for result in results:
                            if result['channel'].lower() == channel_name.lower():
                                video_title = result['title'].lower()
                                if 'cover' not in video_title:
                                    found_videos.append({
                                        'song_dict': song_dict,
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
        except Exception as e:
            continue # Move to next song if error
    return found_videos

def initialize_llm():
    """Initialize the LLM with the current API key"""
    try:
        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7, # Slightly lower for more consistent formatting
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {e}")
        return None

# --- COMPLETELY UPDATED FUNCTION: New prompts for bilingual data ---
def generate_artist_and_songs(genre):
    """
    Generate a distinct artist (with real name) and exactly 3 songs with bilingual titles.
    Returns a dict with 'artist_info' and 'song_list'.
    """
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SequentialChain

    llm = initialize_llm()
    if not llm:
        return None

    # NEW PROMPT: Asks for structured artist data (Stage Name & Real Name)
    prompt_template_artist = PromptTemplate(
        input_variables=["genre"],
        template="""I am trying to discover new music. Give me ONE artist or singer that specializes in {genre}.

        Provide the information in the following EXACT format. Do not add any other text.

        Stage Name: [Artist's stage name or primary performance name]
        Real Name: [Artist's real/full birth name. If primarily known by real name, repeat it. If not publicly known, write "Not widely published".]

        REQUIREMENTS:
        1. The artist must have an active official YouTube channel with music videos.
        2. The real name is CRITICAL for accurate searching.
        3. Use their most recognized stage name."""
    )

    # NEW PROMPT: Asks for 3 songs with original and translated titles
    prompt_template_song = PromptTemplate(
        input_variables=["artist_info"],
        template="""Based on the artist information below, give me EXACTLY 3 of their popular songs that have official music videos.

        Artist Info: {artist_info}

        For EACH song, provide the original title and its English translation/full title.
        Use this EXACT format for your entire response:

        1. Original: [Original song title in native script] | Translation: [English translation/name]
        2. Original: [Original song title in native script] | Translation: [English translation/name]
        3. Original: [Original song title in native script] | Translation: [English translation/name]

        IMPORTANT:
        - Only output the 3 lines above, nothing else.
        - The 'Original' title must be exactly as it appears officially (e.g., 紅蓮華 for LiSA's Gurenge).
        - The 'Translation' should be the common English name or a direct translation (e.g., Gurenge or Crimson Lotus)."""
    )

    # Create chains
    artist_chain = LLMChain(llm=llm, prompt=prompt_template_artist, output_key="artist_raw")
    song_chain = LLMChain(llm=llm, prompt=prompt_template_song, output_key="songs_raw")

    chain = SequentialChain(
        chains=[artist_chain, song_chain],
        input_variables=["genre"],
        output_variables=["artist_raw", "songs_raw"],
        verbose=False
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = chain({"genre": genre})
            # --- PARSE ARTIST INFO ---
            artist_text = response["artist_raw"].strip()
            stage_name, real_name = None, None
            for line in artist_text.split('\n'):
                line = line.strip()
                if line.startswith('Stage Name:'):
                    stage_name = line.replace('Stage Name:', '').strip()
                elif line.startswith('Real Name:'):
                    real_name = line.replace('Real Name:', '').strip()
            if not stage_name:
                continue # Parsing failed, try again
            artist_info = {'stage_name': stage_name, 'real_name': real_name}

            # --- PARSE SONG INFO ---
            songs_text = response["songs_raw"].strip()
            song_list = []
            for line in songs_text.split('\n'):
                line = line.strip()
                # Match lines like "1. Original: ... | Translation: ..."
                if 'Original:' in line and 'Translation:' in line:
                    # Extract the part after the number and period
                    content_part = line.split('.', 1)[-1] if '.' in line else line
                    parts = content_part.split('|')
                    if len(parts) == 2:
                        original = parts[0].replace('Original:', '').strip()
                        translation = parts[1].replace('Translation:', '').strip()
                        song_list.append({'original_title': original, 'english_title': translation})
            # Return only if we successfully parsed exactly 3 songs
            if len(song_list) == 3:
                return {
                    "artist_info": artist_info, # This is now a dict
                    "song_list": song_list      # This is now a list of dicts
                }
        except Exception as e:
            # st.error(f"Attempt {attempt+1} parsing failed: {e}") # Optional debug
            continue
    return None # Failed all attempts

# =============================================================================
# FRONTEND FUNCTIONS - UPDATED
# =============================================================================

def setup_api_key():
    """Handle API key setup"""
    if st.session_state.get('api_key_valid') and st.session_state.get('hide_api_section', False):
        return
    st.sidebar.header("🔑 API Configuration")
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and validate_api_key(env_key):
        os.environ["OPENAI_API_KEY"] = env_key
        st.session_state.api_key_valid = True
        st.sidebar.success("Using API key from environment")
        if 'hide_api_section' not in st.session_state:
            st.session_state.hide_api_section = True
        return
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

# --- UPDATED FUNCTION: Displays bilingual info ---
def display_video_results(artist_info, song_list, found_videos):
    """Display artist info, songs, and found videos in a formatted way."""
    st.subheader(f"🎤 Artist: {artist_info['stage_name']}")
    # Show real name if available and different
    if artist_info['real_name'] and artist_info['real_name'].lower() != artist_info['stage_name'].lower() and artist_info['real_name'] != "Not widely published":
        st.caption(f"**Real Name:** {artist_info['real_name']}")

    st.write("**Selected Songs:**")
    for i, song_dict in enumerate(song_list, 1):
        # Display both original and English title
        display_text = f"{song_dict['original_title']}"
        if song_dict['english_title']:
            display_text += f" ({song_dict['english_title']})"
        st.write(f"{i}. {display_text}")

    if not found_videos:
        st.warning("No videos found from the official channel.")
        return

    st.subheader("🎬 Found Videos")
    cols = st.columns(3)
    for idx, video_info in enumerate(found_videos[:3]):
        with cols[idx % 3]:
            song_dict = video_info['song_dict']
            # Create display title
            display_title = song_dict['original_title']
            if song_dict['english_title']:
                display_title += f"\n({song_dict['english_title']})"
            st.write(f"**{idx+1}. {display_title}**")

            # Video type badge
            type_badges = {
                'official_mv': '🎬 Official MV',
                'live': '🎤 Live',
                'lyric video': '📝 Lyric Video',
                'audio': '🔊 Audio',
                'performance': '🎭 Performance',
                'channel_video': '📺 Channel Video'
            }
            badge = type_badges.get(video_info['type'], video_info['type'])
            st.caption(f"{badge}")

            # Display the video
            video_url = f"https://www.youtube.com/watch?v={video_info['video_id']}"
            st.video(video_url)
            # Show video's actual title
            vid_title = video_info['title']
            if len(vid_title) > 40:
                vid_title = vid_title[:40] + "..."
            st.caption(f"*Video: {vid_title}*")

def main_app():
    """Main application interface - UPDATED LOGIC FLOW"""
    st.title("🎵 I AM MUSIC - Bilingual Artist & Video Finder")
    genre = st.text_input("Enter a music genre:", placeholder="e.g., anime, kpop, rock, hip hop")
    if genre and st.button("Find Artist & Videos", type="primary"):
        with st.spinner("🔍 Finding distinct artist and songs..."):
            # Step 1: Generate artist (with real name) and 3 bilingual songs
            result = generate_artist_and_songs(genre)
            if not result:
                st.error("Could not generate artist and songs. Please try a different genre.")
                return
            artist_info = result["artist_info"]  # dict with 'stage_name', 'real_name'
            song_list = result["song_list"]      # list of dicts with song titles

            # Step 2: Display the generated info
            st.subheader("✅ Recommendation Generated")
            st.write(f"**Stage Name:** {artist_info['stage_name']}")
            if artist_info['real_name'] != "Not widely published":
                st.write(f"**Real Name:** {artist_info['real_name']}")
            st.write("**Songs to Search:**")
            for song in song_list:
                st.caption(f"- {song['original_title']} ({song['english_title']})")

            # Step 3: Find official YouTube channel using REAL NAME
            with st.spinner("Searching for official YouTube channel using real name..."):
                channel_info = find_official_channel(artist_info, genre)
                if channel_info:
                    st.success(f"✅ Found channel: **{channel_info['channel_name']}** (Confidence: {channel_info['confidence']:.0%})")
                else:
                    st.warning("⚠️ Could not definitively find an official channel. Trying a broader search...")
                    # Fallback: search with just the stage name
                    channel_info = find_official_channel({'stage_name': artist_info['stage_name'], 'real_name': artist_info['stage_name']}, genre)

            # Step 4: Search for songs on the identified channel
            if channel_info:
                with st.spinner(f"Searching for songs on **{channel_info['channel_name']}**..."):
                    found_videos = search_songs_on_channel(artist_info, song_list, channel_info)
                # Step 5: Display results
                display_video_results(artist_info, song_list, found_videos)
                # Show channel details in expander
                with st.expander("📺 Channel Details"):
                    st.write(f"**Channel Name:** {channel_info['channel_name']}")
                    if channel_info.get('channel_id'):
                        channel_url = f"https://www.youtube.com/channel/{channel_info['channel_id']}"
                        st.write(f"**Channel URL:** [Click here]({channel_url})")
            else:
                st.error("❌ Could not find a suitable YouTube channel. Please try a different genre.")

# =============================================================================
# APP INITIALIZATION
# =============================================================================

if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'hide_api_section' not in st.session_state:
    st.session_state.hide_api_section = False

setup_api_key()
if not st.session_state.api_key_valid:
    st.info("🔑 Please enter your OpenAI API key in the sidebar to use the app.")
else:
    main_app()
