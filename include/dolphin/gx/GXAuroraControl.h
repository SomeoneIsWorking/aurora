#ifndef DOLPHIN_GXAURORACONTROL_H
#define DOLPHIN_GXAURORACONTROL_H

/**
 * Reserved GX_AURORA_DRAW_TAG payload that marks the NEXT tagged draw as one deforming indexed
 * XYZ-f32 position-array draw. It does not replace the latched identity; send the real tag
 * immediately afterwards.
 */
#define GX_AURORA_DRAW_TAG_INDEXED_DEFORM 0xffffffffffffffffULL

/**
 * Supplies one stable identity per four-position quad for the next indexed-deform draw. Payload:
 * u16 key count followed by that many u64 keys, in vertex-array order. This lets a dynamic batch
 * interpolate surviving quads when births or deaths change the total array length.
 */
#define GX_AURORA_DRAW_INDEXED_KEYS 0x003f

#endif
