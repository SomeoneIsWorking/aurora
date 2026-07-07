#ifndef DOLPHIN_GXGEOMETRY_H
#define DOLPHIN_GXGEOMETRY_H

#include <dolphin/gx/GXEnum.h>

#ifdef __cplusplus
extern "C" {
#endif

void GXSetVtxDesc(GXAttr attr, GXAttrType type);
void GXSetVtxDescv(GXVtxDescList* list);
void GXClearVtxDesc(void);
void GXSetVtxAttrFmtv(GXVtxFmt vtxfmt, const GXVtxAttrFmtList* list);
void GXSetVtxAttrFmt(GXVtxFmt vtxfmt, GXAttr attr, GXCompCnt cnt, GXCompType type, u8 frac);
void GXSetNumTexGens(u8 nTexGens);
void GXBegin(GXPrimitive type, GXVtxFmt vtxfmt, u16 nverts);
#ifdef TARGET_PC
/**
 * Aurora extension: pass as GXBegin's vertex count to have it derived automatically from
 * the number of bytes written before GXEnd. Not supported while recording a display list.
 */
#define GX_AUTO 0xFFFF
#endif
void GXSetTexCoordGen2(GXTexCoordID dst_coord, GXTexGenType func, GXTexGenSrc src_param, u32 mtx, GXBool normalize,
                       u32 postmtx);
void GXSetLineWidth(u8 width, GXTexOffset texOffsets);
void GXSetPointSize(u8 pointSize, GXTexOffset texOffsets);
void GXEnableTexOffsets(GXTexCoordID coord, GXBool line_enable, GXBool point_enable);
#ifdef TARGET_PC
void GXSetArray(GXAttr attr, const void* data, u32 size, u8 stride, bool le);
#define GXSETARRAY(attr, data, size, stride, le) GXSetArray((attr), (data), (size), (stride), (le))
#else
void GXSetArray(GXAttr attr, const void* data, u8 stride);
#define GXSETARRAY(attr, data, size, stride, le) GXSetArray((attr), (data), (stride))
#endif
void GXInvalidateVtxCache(void);

static inline void GXSetTexCoordGen(GXTexCoordID dst_coord, GXTexGenType func, GXTexGenSrc src_param, u32 mtx) {
  GXSetTexCoordGen2(dst_coord, func, src_param, mtx, GX_FALSE, GX_PTIDENTITY);
}

#ifdef __cplusplus
}
#endif

#if defined(TARGET_PC) && defined(__cplusplus)
// GC-original 3-arg GXSetArray signature — decomp code (SMS, SPM, etc.) calls
// this form heavily. `extern "C++"` nested inside the enclosing extern "C" of
// dolphin/gx.h so this can be a C++ overload of the 5-arg TARGET_PC form (C
// linkage doesn't permit overloading). Forwards to the 5-arg form with size=0
// (impl walks the buffer at draw time). Endianness by array kind: the vertex
// attribute arrays (POS..TEX7) carry file-origin GC assets = big-endian; the
// matrix/light arrays (POS_MTX_ARRAY..LIGHT_ARRAY) are runtime-COMPUTED pools
// (J3D draw/normal matrices, light objects) = host-endian on any host.
extern "C++" {
inline void GXSetArray(GXAttr attr, const void* data, u8 stride) {
    const bool le = attr >= GX_POS_MTX_ARRAY && attr <= GX_LIGHT_ARRAY;
    GXSetArray(attr, data, 0u, stride, le);
}
}
#endif

#endif
