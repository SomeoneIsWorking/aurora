#ifndef DOLPHIN_GXAURORACONTROL_H
#define DOLPHIN_GXAURORACONTROL_H

/**
 * Reserved GX_AURORA_DRAW_TAG payload that marks the NEXT tagged draw as one deforming indexed
 * XYZ-f32 position-array draw. It does not replace the latched identity; send the real tag
 * immediately afterwards.
 */
#define GX_AURORA_DRAW_TAG_INDEXED_DEFORM 0xffffffffffffffffULL

#endif
