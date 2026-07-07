#ifndef AURORA_AUDIO_H
#define AURORA_AUDIO_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * aurora::audio — host audio output over an SDL3 audio stream.
 *
 * Push model, designed for a single-threaded game loop: the game mixes one
 * frame's worth of interleaved signed-16-bit samples and pushes them here
 * once per frame. SDL's device layer handles resampling and playback; the
 * caller can pace itself against aurora_audio_queued_frames() to keep the
 * backlog bounded.
 */

/* Open the default playback device for interleaved S16 audio at the given
 * sample rate / channel count. Returns false if the device can't be opened. */
bool aurora_audio_open(uint32_t sample_rate, uint32_t channels);

void aurora_audio_close(void);

/* Queue num_frames interleaved S16 frames (num_frames * channels samples). */
void aurora_audio_push(const int16_t* samples, uint32_t num_frames);

/* Frames pushed but not yet consumed by the device — the pacing backlog. */
uint32_t aurora_audio_queued_frames(void);

#ifdef __cplusplus
}
#endif

#endif
