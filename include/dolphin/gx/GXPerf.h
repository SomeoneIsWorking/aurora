#ifndef DOLPHIN_GXPERF_H
#define DOLPHIN_GXPERF_H

#include <dolphin/types.h>

#ifdef __cplusplus
extern "C" {
#endif

void GXReadXfRasMetric(u32* xf_wait_in, u32* xf_wait_out, u32* ras_busy, u32* clocks);

// GC SDK — draw-sync token API and pixel metric counters.
void GXSetDrawSync(u16 token);
u16 GXReadDrawSync(void);
void GXClearPixMetric(void);
void GXReadPixMetric(u32* top_pixels_in, u32* top_pixels_out,
                     u32* bottom_pixels_in, u32* bottom_pixels_out,
                     u32* clr_pixels_in, u32* clr_pixels_out);

#ifdef __cplusplus
}
#endif

#endif
