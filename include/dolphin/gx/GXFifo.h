#ifndef DOLPHIN_GXFIFO_H
#define DOLPHIN_GXFIFO_H

#include <dolphin/gx/GXEnum.h>

#ifdef __cplusplus
extern "C" {
#endif

// GC SDK — hardware FIFO breakpoint: pause CP when it reads `addr`, resume via
// GXDisableBreakPt. Decomp's TDrawSyncManager uses this to gate draw-sync
// tokens on the GPU.
void GXEnableBreakPt(void* addr);
void GXDisableBreakPt(void);
// GC SDK — CPU-side barrier: block until GPU has processed all pending draw
// commands. Callers use this on the CPU thread that owns the FIFO.
void GXWaitDrawDone(void);

typedef struct {
  u8 pad[128];
} GXFifoObj;

typedef struct OSThread OSThread;

void GXInitFifoBase(GXFifoObj* fifo, void* base, u32 size);
void GXInitFifoPtrs(GXFifoObj* fifo, void* readPtr, void* writePtr);
void GXGetFifoPtrs(GXFifoObj* fifo, void** readPtr, void** writePtr);
OSThread *GXSetCurrentGXThread(void);
OSThread *GXGetCurrentGXThread(void);
GXFifoObj* GXGetCPUFifo(void);
GXFifoObj* GXGetGPFifo(void);
void GXSetCPUFifo(GXFifoObj* fifo);
void GXSetGPFifo(GXFifoObj* fifo);
void GXSaveCPUFifo(GXFifoObj* fifo);
void GXGetFifoStatus(GXFifoObj* fifo, GXBool* overhi, GXBool* underlow, u32* fifoCount, GXBool* cpu_write,
                     GXBool* gp_read, GXBool* fifowrap);
void GXGetGPStatus(GXBool* overhi, GXBool* underlow, GXBool* readIdle, GXBool* cmdIdle, GXBool* brkpt);
void GXInitFifoLimits(GXFifoObj* fifo, u32 hiWaterMark, u32 loWaterMark);
void* GXGetFifoBase(const GXFifoObj* fifo);
u32 GXGetFifoSize(const GXFifoObj* fifo);

#ifdef __cplusplus
}
#endif

#endif
