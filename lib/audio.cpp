#include <aurora/audio.h>

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_init.h>

#include "internal.hpp"

namespace {
aurora::Module Log("aurora::audio");

SDL_AudioStream* g_stream = nullptr;
uint32_t g_channels = 0;
uint32_t g_sampleRate = 0;
} // namespace

extern "C" {

bool aurora_audio_open(uint32_t sample_rate, uint32_t channels) {
  if (g_stream != nullptr) {
    Log.warn("aurora_audio_open: already open ({} Hz, {} ch)", g_sampleRate, g_channels);
    return true;
  }
  if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
    Log.error("SDL_InitSubSystem(AUDIO) failed: {}", SDL_GetError());
    return false;
  }
  const SDL_AudioSpec spec{
      .format = SDL_AUDIO_S16,
      .channels = static_cast<int>(channels),
      .freq = static_cast<int>(sample_rate),
  };
  g_stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
  if (g_stream == nullptr) {
    Log.error("SDL_OpenAudioDeviceStream failed: {}", SDL_GetError());
    return false;
  }
  g_channels = channels;
  g_sampleRate = sample_rate;
  SDL_ResumeAudioStreamDevice(g_stream);
  Log.info("audio out open: {} Hz, {} ch (S16)", sample_rate, channels);
  return true;
}

void aurora_audio_close(void) {
  if (g_stream == nullptr) {
    return;
  }
  SDL_DestroyAudioStream(g_stream);
  g_stream = nullptr;
  g_channels = 0;
  g_sampleRate = 0;
}

void aurora_audio_push(const int16_t* samples, uint32_t num_frames) {
  if (g_stream == nullptr || samples == nullptr || num_frames == 0) {
    return;
  }
  const int bytes = static_cast<int>(num_frames * g_channels * sizeof(int16_t));
  if (!SDL_PutAudioStreamData(g_stream, samples, bytes)) {
    Log.error("SDL_PutAudioStreamData failed: {}", SDL_GetError());
  }
}

uint32_t aurora_audio_queued_frames(void) {
  if (g_stream == nullptr || g_channels == 0) {
    return 0;
  }
  const int bytes = SDL_GetAudioStreamQueued(g_stream);
  if (bytes <= 0) {
    return 0;
  }
  return static_cast<uint32_t>(bytes) / (g_channels * sizeof(int16_t));
}

} // extern "C"
