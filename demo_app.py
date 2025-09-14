import ascui
import os
import math
import pygame
import threading
import yt_dlp
from pinqy import p
from PyQt6.QtCore import Qt
import time
import random
import numpy as np

# setup
MP3_DIR = "mp3"
if not os.path.exists(MP3_DIR):
    os.makedirs(MP3_DIR)
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)


def get_initial_state():
    return {
        'current_page': 'dashboard', 'focus': '1: Dashboard',
        'songs_in_library': len(p(os.listdir(MP3_DIR)).where(lambda f: f.endswith('.mp3')).to.list()),
        'search_query': '', 'search_status': 'idle', 'search_results': [], 'selected_result_idx': 0,
        'library_songs': [], 'selected_song_idx': 0, 'playback_status': 'stopped', 'now_playing': None,
        'volume': 0.7, 'playback_position': 0, 'song_length': 0,
        'viz_buffer': np.zeros(512, dtype=np.float32), 'viz_freqs': [0.0] * 32,
        'viz_phase': 0.0, 'viz_energy': 0.0, 'viz_smoothed_freqs': [0.0] * 32,
        'current_sound': None, 'sound_channel': None, 'song_start_time': 0
    }


def update_state(state, action):
    action_type, payload = action

    focus_map = {
        'dashboard': ['1: Dashboard', '2: Search', '3: Library', 'Quit'],
        'search': ['1: Dashboard', '2: Search', '3: Library', 'Quit', 'search_input', 'search_results_list',
                   'Download'],
        'library': ['1: Dashboard', '2: Search', '3: Library', 'Quit', 'song_list', 'Play', 'Stop', 'Volume+',
                    'Volume-']
    }

    def handle_focus_change(direction):
        page_focuses = focus_map.get(state['current_page'], [])
        if not page_focuses: return state
        try:
            current_idx = page_focuses.index(state['focus'])
            state['focus'] = page_focuses[(current_idx + direction) % len(page_focuses)]
        except ValueError:
            state['focus'] = page_focuses[0]
        return state

    if action_type == 'switch_page':
        state['current_page'] = payload
        if payload == 'library':
            state['library_songs'] = p(os.listdir(MP3_DIR)).where(lambda f: f.endswith('.mp3')).to.list()
        state['focus'] = focus_map[payload][4] if len(focus_map[payload]) > 4 else focus_map[payload][0]

    elif action_type == 'query_change':
        state['search_query'] = payload

    elif action_type == 'select_result_up':
        state['selected_result_idx'] = max(0, state['selected_result_idx'] - 1)

    elif action_type == 'select_result_down':
        state['selected_result_idx'] = min(len(state['search_results']) - 1, state['selected_result_idx'] + 1)

    elif action_type == 'execute_search':
        if not state['search_query'].strip(): return state
        state['search_status'] = 'searching...'
        state['search_results'] = []
        state['selected_result_idx'] = 0

        def search_in_thread():
            try:
                ydl_opts = {
                    'format': 'bestaudio',
                    'noplaylist': True,
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'ignoreerrors': True,
                    'socket_timeout': 30
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    search_results = ydl.extract_info(f"ytsearch5:{state['search_query']}", download=False)
                    if search_results and 'entries' in search_results:
                        valid_entries = [entry for entry in search_results['entries'] if entry is not None]
                        state['search_results'] = valid_entries
                        state['search_status'] = f"found {len(valid_entries)} results"
                    else:
                        state['search_status'] = 'no results found'
            except Exception as e:
                print(f"Search error: {e}")
                state['search_status'] = f'search failed: {str(e)[:50]}'

        threading.Thread(target=search_in_thread, daemon=True).start()

    elif action_type == 'download_video':
        if not state['search_results'] or state['selected_result_idx'] >= len(state['search_results']):
            return state

        video = state['search_results'][state['selected_result_idx']]
        title = video.get('title', 'unknown')[:30]
        video_id = video.get('id')

        if not video_id:
            state['search_status'] = 'download failed: no video id'
            return state

        state['search_status'] = f"downloading: {title}..."

        def download():
            try:
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"

                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(MP3_DIR, '%(title)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                state['search_status'] = f'downloaded: {title}'
                state['songs_in_library'] = len(p(os.listdir(MP3_DIR)).where(lambda f: f.endswith('.mp3')).to.list())
            except Exception as e:
                print(f"Download error: {e}")
                state['search_status'] = f'download failed: {str(e)[:30]}'

        threading.Thread(target=download, daemon=True).start()

    elif action_type == 'select_song_up':
        state['selected_song_idx'] = max(0, state['selected_song_idx'] - 1)

    elif action_type == 'select_song_down':
        state['selected_song_idx'] = min(len(state['library_songs']) - 1, state['selected_song_idx'] + 1)

    elif action_type == 'play_song':
        if not state['library_songs'] or state['selected_song_idx'] >= len(state['library_songs']):
            return state
        song = state['library_songs'][state['selected_song_idx']]
        try:
            pygame.mixer.music.load(os.path.join(MP3_DIR, song))
            pygame.mixer.music.set_volume(state['volume'])
            pygame.mixer.music.play()
            state['playback_status'] = 'playing'
            state['now_playing'] = song
            state['song_start_time'] = time.time()
        except Exception as e:
            state['playback_status'] = f'error: {str(e)[:20]}'

    elif action_type == 'stop_song':
        pygame.mixer.music.stop()
        state['playback_status'] = 'stopped'

    elif action_type == 'volume_up':
        state['volume'] = min(1.0, state['volume'] + 0.1)
        pygame.mixer.music.set_volume(state['volume'])

    elif action_type == 'volume_down':
        state['volume'] = max(0.0, state['volume'] - 0.1)
        pygame.mixer.music.set_volume(state['volume'])

    elif action_type == 'focus_next':
        return handle_focus_change(1)
    elif action_type == 'focus_prev':
        return handle_focus_change(-1)
    elif action_type == 'quit':
        pygame.quit()
        exit()

    update_audio_analysis(state)
    return state


def update_audio_analysis(state):
    t = time.time()

    if pygame.mixer.music.get_busy():
        song_pos = (t - state.get('song_start_time', 0)) % 60

        # generate realistic frequency spectrum based on song position
        freqs = []
        for i in range(32):
            freq_hz = 20 * (2 ** (i / 4))

            # create complex musical patterns
            bass_weight = 1.0 if i < 8 else 0.3
            mid_weight = 1.0 if 8 <= i < 20 else 0.4
            high_weight = 1.0 if i >= 20 else 0.2

            # multiple overlapping rhythmic patterns
            beat_1 = math.sin(song_pos * 2.4 + i * 0.3) ** 2
            beat_2 = math.sin(song_pos * 3.6 + i * 0.1) ** 2
            beat_3 = math.sin(song_pos * 1.8 + i * 0.2) ** 2

            # harmonic content with phase relationships
            harmonic = (
                    0.8 * math.sin(song_pos * 4.2 + i * 0.4 + math.pi * 0.3) +
                    0.5 * math.sin(song_pos * 8.4 + i * 0.2 + math.pi * 0.6) +
                    0.3 * math.sin(song_pos * 16.8 + i * 0.1 + math.pi * 0.9)
            )

            # frequency-dependent modulation
            freq_mod = 0.5 + 0.5 * math.sin(freq_hz * 0.001 + song_pos * 0.8)

            # combine all components
            base_amplitude = bass_weight * beat_1 + mid_weight * beat_2 + high_weight * beat_3
            final_amp = (base_amplitude + harmonic * 0.3) * freq_mod * state.get('volume', 0.7)

            freqs.append(max(0, min(1, final_amp * 0.5 + 0.1)))

        # smooth transitions with different rates per frequency band
        smoothed = state.get('viz_smoothed_freqs', [0.0] * 32)
        for i in range(32):
            # bass frequencies are smoothed less (more responsive)
            alpha = 0.85 if i < 8 else 0.70 if i < 20 else 0.60
            smoothed[i] = alpha * freqs[i] + (1 - alpha) * smoothed[i]

        state['viz_freqs'] = freqs
        state['viz_smoothed_freqs'] = smoothed
        state['viz_energy'] = p(freqs).stats.average()
        state['viz_phase'] += state['viz_energy'] * 0.2

    else:
        decay = 0.88
        state['viz_freqs'] = [f * decay for f in state.get('viz_freqs', [0] * 32)]
        state['viz_smoothed_freqs'] = [f * decay for f in state.get('viz_smoothed_freqs', [0] * 32)]
        state['viz_energy'] *= decay


def waveform_mandala(freqs, w, h, t):
    chars = " ·`'^~*+=xX%@▓█"
    cx, cy = w // 2, h // 2

    return p(range(h)).select(lambda y:
                              "".join(p(range(w)).select(lambda x:
                                                         (lambda dx, dy, r, theta:
                                                          (lambda freq_idx:
                                                           (lambda wave_val:
                                                            chars[
                                                                max(0, min(len(chars) - 1, int(wave_val * len(chars))))]
                                                            )(
                                                               freqs[freq_idx] *
                                                               abs(math.sin(r * 0.8 + theta * 3 + t * 4 + freqs[
                                                                   freq_idx % 8] * 10)) *
                                                               (1 - r / max(w, h)) *
                                                               abs(math.cos(theta * freqs[
                                                                   (freq_idx + 4) % len(freqs)] * 8 + t * 3))
                                                           )
                                                           )(int((r + theta * 2) % len(freqs)))
                                                          )(x - cx, y - cy, math.sqrt((x - cx) ** 2 + (y - cy) ** 2),
                                                            math.atan2(y - cy, x - cx))
                                                         ).to.list())
                              ).to.list()


def spiral_galaxy(freqs, w, h, t):
    chars = " ·`'^*+=xX%@▓█"
    cx, cy = w // 2, h // 2

    # extract dominant frequency characteristics
    bass_power = p(freqs[0:8]).stats.sum()
    mid_power = p(freqs[8:20]).stats.sum()
    high_power = p(freqs[20:]).stats.sum()

    spiral_speed = bass_power * 2.0
    spiral_arms = max(2, int(mid_power * 8))
    brightness = high_power * 1.5

    return p(range(h)).select(lambda y:
                              "".join(p(range(w)).select(lambda x:
                                                         (lambda dx, dy, r, theta:
                                                          (lambda spiral_theta, arm_intensity:
                                                           (lambda noise_val:
                                                            chars[max(0, min(len(chars) - 1, int(
                                                                arm_intensity * brightness * noise_val * len(chars)
                                                            )))]
                                                            )(
                                                               abs(math.sin(r * 0.3 + spiral_theta + t * 2)) *
                                                               abs(math.cos(
                                                                   theta * freqs[int(r) % len(freqs)] * 4 + t * 6))
                                                           )
                                                           )(
                                                              theta + r * 0.2 + t * spiral_speed,
                                                              max(0, 1 - abs(
                                                                  math.sin(theta * spiral_arms + r * 0.1 + t * 3)) * 2)
                                                          )
                                                          )(x - cx, y - cy, math.sqrt((x - cx) ** 2 + (y - cy) ** 2),
                                                            math.atan2(y - cy, x - cx))
                                                         ).to.list())
                              ).to.list()


def neural_network(freqs, w, h, t):
    chars = " ·`'^*+=#@▓█"

    # create nodes based on frequency peaks
    nodes = p(range(len(freqs))).where(lambda i: freqs[i] > 0.3).select(lambda i: {
        'x': (i * w // len(freqs) + int(freqs[i] * w * 0.2 * math.sin(t * 2 + i))) % w,
        'y': int(h * 0.3 + freqs[i] * h * 0.4 * math.cos(t * 1.5 + i * 0.3)),
        'strength': freqs[i],
        'freq_idx': i
    }).to.list()

    return p(range(h)).select(lambda y:
                              "".join(p(range(w)).select(lambda x:
                                                         (lambda connections:
                                                          chars[max(0,
                                                                    min(len(chars) - 1, int(connections * len(chars))))]
                                                          )(
                                                             p(nodes).select(lambda node:
                                                                             (lambda dist:
                                                                              node['strength'] *
                                                                              math.exp(-dist * 0.1) *
                                                                              abs(math.sin(dist * 0.5 + t * 3 + freqs[
                                                                                  node['freq_idx']] * 8))
                                                                              )(math.sqrt((x - node['x']) ** 2 + (
                                                                                         y - node['y']) ** 2))
                                                                             ).stats.sum()
                                                         )
                                                         ).to.list())
                              ).to.list()


def crystal_lattice(freqs, w, h, t):
    chars = " ·`'^*+=xX%@▓█"

    # frequency-driven crystal parameters
    lattice_scale = 2 + p(freqs[0:8]).stats.average() * 6
    refraction = p(freqs[8:16]).stats.average() * 2
    resonance = p(freqs[16:24]).stats.sum()

    return p(range(h)).select(lambda y:
                              "".join(p(range(w)).select(lambda x:
                                                         (lambda lattice_x, lattice_y:
                                                          (lambda crystal_val:
                                                           chars[max(0, min(len(chars) - 1,
                                                                            int(crystal_val * len(chars))))]
                                                           )(
                                                              abs(math.sin(lattice_x + refraction * math.cos(
                                                                  lattice_y + t * 2))) *
                                                              abs(math.cos(lattice_y + refraction * math.sin(
                                                                  lattice_x + t * 1.5))) *
                                                              (0.5 + 0.5 * math.sin(
                                                                  math.sqrt(lattice_x ** 2 + lattice_y ** 2) * 0.5 +
                                                                  t * 4 + resonance * 10
                                                              ))
                                                          )
                                                          )(x * lattice_scale / w, y * lattice_scale / h)
                                                         ).to.list())
                              ).to.list()


def plasma_field(freqs, w, h, t):
    chars = " ·`'^~*+=xX%@▓█"

    # multi-layered plasma with frequency modulation
    field_strength = p(freqs).stats.average()
    turbulence = p(freqs).stats.std_dev() * 2

    return p(range(h)).select(lambda y:
                              "".join(p(range(w)).select(lambda x:
                                                         (lambda nx, ny:
                                                          (lambda plasma_layers:
                                                           chars[max(0, min(len(chars) - 1,
                                                                            int(plasma_layers * len(chars))))]
                                                           )(
                                                              # multiple interference patterns
                                                              abs(math.sin(nx * 2 + t * 3 + freqs[0] * 8)) *
                                                              abs(math.cos(ny * 1.5 + t * 2.5 + freqs[4] * 6)) +
                                                              0.7 * abs(
                                                                  math.sin(nx * 3 + ny * 2 + t * 4 + freqs[8] * 10)) +
                                                              0.5 * abs(
                                                                  math.cos(nx * 4 + ny * 3 + t * 5 + freqs[12] * 12)) +
                                                              turbulence * abs(
                                                                  math.sin(nx * 8 + ny * 6 + t * 8)) * field_strength
                                                          ) * 0.25
                                                          )(x * 0.3, y * 0.3)
                                                         ).to.list())
                              ).to.list()


def quantum_foam(freqs, w, h, t):
    chars = " ·`'^~*+=xX%@▓█"

    # quantum-scale fluctuations driven by high frequencies
    vacuum_energy = p(freqs[16:]).stats.average()
    coherence_length = 3 + p(freqs[0:8]).stats.sum() * 4
    entanglement = p(freqs).stats.sum() * 0.1

    return p(range(h)).select(lambda y:
                              "".join(p(range(w)).select(lambda x:
                                                         (lambda quantum_state:
                                                          chars[max(0, min(len(chars) - 1,
                                                                           int(abs(quantum_state) * len(chars))))]
                                                          )(
                                                             # superposition of quantum states
                                                             math.sin(
                                                                 x * coherence_length + t * 6 + entanglement * 20) *
                                                             math.cos(
                                                                 y * coherence_length + t * 4 + entanglement * 15) *
                                                             abs(math.sin(x * y * 0.1 + t * 8 + vacuum_energy * 25)) *
                                                             (0.7 + 0.3 * math.sin(
                                                                 math.sqrt(x ** 2 + y ** 2) * 0.2 + t * 10))
                                                         )
                                                         ).to.list())
                              ).to.list()


def viz(t, state):
    w, h = 45, 14
    freqs = state.get('viz_smoothed_freqs', [0.0] * 32)
    energy = state.get('viz_energy', 0.0)

    if energy < 0.02:
        return [" " * w for _ in range(h)]

    # dynamic mode selection based on spectral characteristics
    bass_dom = p(freqs[0:8]).stats.average()
    mid_dom = p(freqs[8:20]).stats.average()
    high_dom = p(freqs[20:]).stats.average()
    spectral_balance = bass_dom + mid_dom * 2 + high_dom * 3
    tempo_estimate = p(freqs[0:4]).stats.std_dev() * 20

    # mode selection influenced by musical characteristics
    mode_val = (spectral_balance * 10 + tempo_estimate + t * energy * 2) % 6
    mode = int(mode_val)

    try:
        if mode == 0:
            return waveform_mandala(freqs, w, h, t)
        elif mode == 1:
            return spiral_galaxy(freqs, w, h, t)
        elif mode == 2:
            return neural_network(freqs, w, h, t)
        elif mode == 3:
            return crystal_lattice(freqs, w, h, t)
        elif mode == 4:
            return plasma_field(freqs, w, h, t)
        else:
            return quantum_foam(freqs, w, h, t)
    except:
        # emergency fallback using raw frequency data
        chars = " ·*@█"
        return p(range(h)).select(lambda y:
                                  "".join(p(range(w)).select(lambda x:
                                                             chars[int(freqs[(x + y) % len(freqs)] * len(chars)) % len(
                                                                 chars)]
                                                             ).to.list())
                                  ).to.list()


def dashboard_view(state):
    status_color = (100, 255, 100) if state['playback_status'] == 'playing' else (200, 200, 200)

    return ascui.v(
        ascui.txt("ASCII YouTube Downloader & Player").with_style(fg=(100, 255, 100)),
        ascui.spacer(),
        ascui.txt(f"Music Library: {state['songs_in_library']} songs"),
        ascui.txt(f"Currently: {state['playback_status']}").with_style(fg=status_color),
        ascui.txt(f"Volume: {int(state['volume'] * 100)}%"),
        ascui.spacer(),
        ascui.txt("Navigate with [1] Search [2] Dashboard [3] Library"),
        ascui.txt("Search YouTube and download music, then play from your library"),
        ascui.spacer(),
        ascui.txt("Now Playing:").with_style(fg=(255, 255, 100)) if state.get('now_playing') else ascui.txt(""),
        ascui.txt(state.get('now_playing', '')[:-4] if state.get('now_playing') else '').with_style(
            fg=(255, 255, 100)) if state.get('now_playing') else ascui.txt(""),
    ).with_style(padding=(2, 2, 2, 2))


def search_view(state):
    selected_result = None
    if state['search_results'] and 0 <= state['selected_result_idx'] < len(state['search_results']):
        selected_result = state['search_results'][state['selected_result_idx']]

    status_colors = {
        'searching...': (255, 255, 100),
        'idle': (150, 150, 150)
    }

    if state['search_status'].startswith('found'):
        status_color = (100, 255, 100)
    elif state['search_status'].startswith('downloaded'):
        status_color = (100, 255, 100)
    elif 'failed' in state['search_status'] or 'no results' in state['search_status']:
        status_color = (255, 100, 100)
    else:
        status_color = status_colors.get(state['search_status'], (150, 150, 150))

    formatted_results = []
    for i, result in enumerate(state['search_results']):
        title = result.get('title', 'No Title')[:50]
        duration = result.get('duration_string', 'Unknown')
        uploader = result.get('uploader', 'Unknown')[:20]
        formatted_results.append(f"{title:<50} │ {duration:<8} │ {uploader}")

    return ascui.h(
        ascui.v(
            ascui.txt("YouTube Search").with_style(fg=(255, 100, 100)),
            ascui.spacer(),
            ascui.inp(
                state['search_query'],
                "Enter search query and press Enter...",
                on_change=lambda v: ('query_change', v)
            ).with_key('search_input'),
            ascui.spacer(),
            ascui.txt(state['search_status']).with_style(fg=status_color),
            ascui.spacer(),
            ascui.txt("Results:") if state['search_results'] else ascui.txt(""),
            ascui.lst(
                formatted_results,
                state['selected_result_idx'],
                lambda r: r
            ).with_key('search_results_list').grow() if state['search_results'] else ascui.txt("").grow()
        ).grow().with_style(padding=(1, 1, 1, 1)),

        ascui.v(
            ascui.txt("Video Details").with_style(fg=(100, 200, 255)),
            ascui.spacer(),
            ascui.txt(f"Title: {selected_result.get('title', 'N/A')[:40]}...") if selected_result else ascui.txt(
                "Select a video"),
            ascui.txt(f"Channel: {selected_result.get('uploader', 'N/A')}") if selected_result else ascui.txt(""),
            ascui.txt(f"Duration: {selected_result.get('duration_string', 'N/A')}") if selected_result else ascui.txt(
                ""),
            ascui.txt(f"Views: {selected_result.get('view_count', 0):,}") if selected_result and selected_result.get(
                'view_count') else ascui.txt(""),
            ascui.spacer(),
            ascui.btn("Download", "download_video").with_key("Download") if selected_result else ascui.txt("")
        ).with_style(padding=(1, 1, 1, 1)).grow()
    )


def library_view(state):
    formatted_songs = p(state['library_songs']).select(
        lambda song: song.rsplit('.', 1)[0]
    ).to.list()

    volume_bar = "█" * int(state['volume'] * 10) + "░" * (10 - int(state['volume'] * 10))

    # frequency visualization
    freqs = state.get('viz_smoothed_freqs', [0.0] * 32)
    freq_bars = p(range(0, 24, 3)).select(lambda i:
                                          "█" * max(1, int(freqs[i] * 6)) if i < len(freqs) else ""
                                          ).to.list()

    return ascui.h(
        ascui.v(
            ascui.txt("Music Library").with_style(fg=(100, 255, 100)),
            ascui.spacer(),
            ascui.lst(
                formatted_songs,
                state['selected_song_idx'],
                lambda s: s
            ).with_key('song_list').grow(),
            ascui.spacer(),
            ascui.h(
                ascui.btn("Play", "play_song").with_key("Play"),
                ascui.btn("Stop", "stop_song").with_key("Stop"),
                ascui.btn("Vol+", "volume_up").with_key("Volume+"),
                ascui.btn("Vol-", "volume_down").with_key("Volume-")
            )
        ).grow().with_style(padding=(1, 1, 1, 1)),

        ascui.v(
            ascui.txt("Now Playing").with_style(fg=(255, 255, 100)),
            ascui.txt(state.get('now_playing', 'Nothing playing').rsplit('.', 1)[0] if state.get(
                'now_playing') else 'Nothing playing').with_style(fg=(200, 200, 200)),
            ascui.txt(f"Status: {state['playback_status']}"),
            ascui.txt(f"Volume: {volume_bar} {int(state['volume'] * 100)}%"),
            ascui.txt(f"Energy: {'█' * int(state.get('viz_energy', 0) * 20)}"),
            ascui.spacer(),
            ascui.txt("Psychedelic Audio Visualizer").with_style(fg=(100, 200, 255)),
            ascui.txt("Modes: Mandala→Galaxy→Neural→Crystal→Plasma→Quantum"),
            ascui.txt(f"Spectrum: {' '.join(freq_bars)}").with_style(fg=(255, 150, 255)),
            ascui.spacer(),
            ascui.anim(frames_fn=lambda t: viz(t, state)).grow()
        ).with_style(padding=(1, 1, 1, 1)).grow()
    )

def main_view(state):
    pages = {
        'dashboard': dashboard_view,
        'search': search_view,
        'library': library_view
    }

    help_text = "[Tab]/[Shift+Tab] Focus │ [Enter] Select/Search │ [↑/↓] Navigate │ [1-3] Switch Pages │ [Esc] Quit"

    return ascui.v(
        ascui.h(
            ascui.btn("1: Dashboard", "switch_page", "dashboard").with_style(
                bg=(80, 80, 120) if state['current_page'] == 'dashboard' else None
            ).with_key("1: Dashboard"),
            ascui.btn("2: Search", "switch_page", "search").with_style(
                bg=(80, 80, 120) if state['current_page'] == 'search' else None
            ).with_key("2: Search"),
            ascui.btn("3: Library", "switch_page", "library").with_style(
                bg=(80, 80, 120) if state['current_page'] == 'library' else None
            ).with_key("3: Library"),
            ascui.spacer(),
            ascui.btn("Quit", "quit").with_key("Quit")
        ).with_style(padding=(1, 1, 0, 1)),

        pages[state['current_page']](state).grow().with_style(border=ascui.borders.rounded),

        ascui.txt(
            f"Focus: {state['focus']} │ Page: {state['current_page']} │ Songs: {state['songs_in_library']}").with_style(
            fg=(150, 150, 150), padding=(1, 1, 0, 1)
        ),
        ascui.txt(help_text).with_style(
            bg=(45, 35, 55), fg=(180, 180, 200), padding=(1, 1, 1, 1)
        )
    )


if __name__ == "__main__":
    keymap = {
        Qt.Key.Key_Up: 'nav_up',
        Qt.Key.Key_Down: 'nav_down',
        '1': ('switch_page', 'dashboard'),
        '2': ('switch_page', 'search'),
        '3': ('switch_page', 'library'),
        Qt.Key.Key_Escape: 'quit',
        Qt.Key.Key_Enter: 'Key_Enter',
        Qt.Key.Key_Return: 'Key_Enter',
        '+': 'volume_up',
        '-': 'volume_down'
    }


    def custom_update(state, action):
        action_type, payload = action
        focus = state.get('focus', '')

        if focus == 'search_results_list' and action_type == 'nav_up':
            action = ('select_result_up', None)
        elif focus == 'search_results_list' and action_type == 'nav_down':
            action = ('select_result_down', None)
        elif focus == 'song_list' and action_type == 'nav_up':
            action = ('select_song_up', None)
        elif focus == 'song_list' and action_type == 'nav_down':
            action = ('select_song_down', None)
        elif action_type == 'Key_Enter' and focus == 'search_input':
            action = ('execute_search', None)
        elif action_type == 'Key_Enter' and focus == 'song_list':
            action = ('play_song', None)
        elif action_type == 'Key_Enter' and focus == 'Download':
            action = ('download_video', None)
        elif action_type == 'volume_up' and focus in ['Volume+', 'song_list']:
            action = ('volume_up', None)
        elif action_type == 'volume_down' and focus in ['Volume-', 'song_list']:
            action = ('volume_down', None)

        return update_state(state, action)


    main_page = ascui.page(
        view_fn=main_view,
        update_fn=custom_update,
        keymap=keymap
    )

    (ascui.app("ASCII YouTube Downloader", 1400, 900)
     .with_state(get_initial_state())
     .with_theme(ascui.dark_theme())
     .add_page("main", main_page)
     .run())