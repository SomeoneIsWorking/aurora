include(FetchContent)
include(GoogleTest)

if (NOT TARGET gtest)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(googletest)
endif ()

if (AURORA_ENABLE_GX)
  # GX FIFO round-trip tests
  # Compile GX sources directly to avoid pulling in the full renderer/WebGPU runtime
  add_executable(gx_fifo_tests
    gx_fifo_test.cpp
    gx_test_stubs.cpp
    # GX API implementations (encoders)
    ../lib/dolphin/gx/GXBump.cpp
    ../lib/dolphin/gx/GXCull.cpp
    ../lib/dolphin/gx/GXCpu2Efb.cpp
    ../lib/dolphin/gx/GXDispList.cpp
    ../lib/dolphin/gx/GXDraw.cpp
    ../lib/dolphin/gx/GXExtra.cpp
    ../lib/dolphin/gx/GXFifo.cpp
    ../lib/dolphin/gx/GXFrameBuffer.cpp
    ../lib/dolphin/gx/GXGeometry.cpp
    ../lib/dolphin/gx/GXGet.cpp
    ../lib/dolphin/gx/GXLighting.cpp
    ../lib/dolphin/gx/GXManage.cpp
    ../lib/dolphin/gx/GXAurora.cpp
    ../lib/dolphin/gx/GXPerf.cpp
    ../lib/dolphin/gx/GXPixel.cpp
    ../lib/dolphin/gx/GXTev.cpp
    ../lib/dolphin/gx/GXTexture.cpp
    ../lib/dolphin/gx/GXTransform.cpp
    ../lib/dolphin/gx/GXVert.cpp
    # FIFO/command processor (encoder + decoder)
    ../lib/gx/fifo.cpp
    ../lib/gx/command_processor.cpp
    # Display list reader/optimizer
    ../lib/gx/attr_fmt.cpp
    ../lib/gx/dl.cpp
    gx_dl_test.cpp
  )

  target_include_directories(gx_fifo_tests PRIVATE
    ../include
    ../lib
    ../lib/dolphin/gx
  )

  target_compile_definitions(gx_fifo_tests PRIVATE AURORA TARGET_PC)

  target_link_libraries(gx_fifo_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    lucent::lucent
    xxhash
    absl::flat_hash_map
    absl::btree
    dawn::dawncpp_headers
    TracyClient
    ${AURORA_SDL3_TARGET}
  )

  gtest_discover_tests(gx_fifo_tests)

  add_executable(render_worker_tests
    render_worker_test.cpp
    os_test_globals.cpp
    ../lib/logging.cpp
    ../lib/gfx/render_worker.cpp
  )
  target_include_directories(render_worker_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(render_worker_tests PRIVATE AURORA TARGET_PC)
  target_link_libraries(render_worker_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    TracyClient
  )
  gtest_discover_tests(render_worker_tests)

  add_executable(persistent_upload_tests
    persistent_upload_test.cpp
    os_test_globals.cpp
    ../lib/logging.cpp
    ../lib/gfx/persistent_upload.cpp
    ../lib/gfx/render_worker.cpp
  )
  target_include_directories(persistent_upload_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(persistent_upload_tests PRIVATE AURORA TARGET_PC)
  target_link_libraries(persistent_upload_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    TracyClient
  )
  gtest_discover_tests(persistent_upload_tests)

  add_executable(replay_draw_validation_tests
    replay_draw_validation_test.cpp
    os_test_globals.cpp
    ../lib/logging.cpp
    ../lib/gfx/replay_draw_validation.cpp
  )
  target_include_directories(replay_draw_validation_tests PRIVATE
    ../include
    ../lib
  )
  target_compile_definitions(replay_draw_validation_tests PRIVATE AURORA TARGET_PC)
  target_link_libraries(replay_draw_validation_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    dawn::dawncpp_headers
    TracyClient
    xxhash
    absl::flat_hash_map
  )
  gtest_discover_tests(replay_draw_validation_tests)

  add_executable(frame_readback_flight_tests frame_readback_flight_test.cpp)
  target_include_directories(frame_readback_flight_tests PRIVATE ../lib)
  target_link_libraries(frame_readback_flight_tests PRIVATE gtest gtest_main)
  gtest_discover_tests(frame_readback_flight_tests)

  add_executable(frame_sink_schedule_tests frame_sink_schedule_test.cpp)
  target_include_directories(frame_sink_schedule_tests PRIVATE ../include ../lib)
  target_link_libraries(frame_sink_schedule_tests PRIVATE gtest gtest_main)
  gtest_discover_tests(frame_sink_schedule_tests)

  add_executable(texture_usage_tests texture_usage_test.cpp)
  target_include_directories(texture_usage_tests PRIVATE ../lib)
  target_link_libraries(texture_usage_tests PRIVATE gtest gtest_main dawn::dawncpp_headers)
  gtest_discover_tests(texture_usage_tests)

  add_executable(pipeline_worker_gate_tests pipeline_worker_gate_test.cpp)
  target_include_directories(pipeline_worker_gate_tests PRIVATE ../lib)
  target_link_libraries(pipeline_worker_gate_tests PRIVATE gtest gtest_main)
  gtest_discover_tests(pipeline_worker_gate_tests)

  add_executable(gpu_submit_probe_tests
    gpu_submit_probe_test.cpp
    os_test_globals.cpp
    ../lib/gfx/gpu_submit_probe.cpp
    ../lib/logging.cpp
  )
  target_include_directories(gpu_submit_probe_tests PRIVATE ../include ../lib)
  target_compile_definitions(gpu_submit_probe_tests PRIVATE AURORA TARGET_PC WEBGPU_DAWN)
  target_link_libraries(gpu_submit_probe_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    xxhash
    dawn::dawncpp_headers
    TracyClient
  )
  gtest_discover_tests(gpu_submit_probe_tests)

  add_executable(replay_lineage_tests
    replay_lineage_test.cpp
    os_test_globals.cpp
    ../lib/gfx/gpu_submit_probe.cpp
    ../lib/gfx/replay_lineage.cpp
    ../lib/logging.cpp
  )
  target_include_directories(replay_lineage_tests PRIVATE ../include ../lib)
  target_compile_definitions(replay_lineage_tests PRIVATE AURORA TARGET_PC WEBGPU_DAWN)
  target_link_libraries(replay_lineage_tests PRIVATE
    gtest
    gtest_main
    fmt::fmt
    xxhash
    dawn::dawncpp_headers
    TracyClient
  )
  gtest_discover_tests(replay_lineage_tests)

  add_executable(debug_markers_tests
    debug_markers_test.cpp
    os_test_globals.cpp
    ../lib/gfx/debug_markers.cpp
    ../lib/logging.cpp
  )
  target_include_directories(debug_markers_tests PRIVATE ../include ../lib)
  target_compile_definitions(debug_markers_tests PRIVATE AURORA TARGET_PC)
  target_link_libraries(debug_markers_tests PRIVATE gtest gtest_main fmt::fmt TracyClient)
  gtest_discover_tests(debug_markers_tests)

  add_executable(debug_group_snapshots_tests debug_group_snapshots_test.cpp)
  target_link_libraries(debug_group_snapshots_tests PRIVATE gtest gtest_main)
  gtest_discover_tests(debug_group_snapshots_tests)

  # Triangle-index generation. Header-only, so this needs no GX/WebGPU link — which is the point:
  # the code it covers used to be static inside command_processor.cpp and therefore untestable.
  add_executable(gx_prim_index_tests gx_prim_index_test.cpp)
  target_link_libraries(gx_prim_index_tests PRIVATE gtest gtest_main)
  gtest_discover_tests(gx_prim_index_tests)

  add_executable(gx_auto_array_sizing_tests gx_auto_array_sizing_test.cpp)
  target_link_libraries(gx_auto_array_sizing_tests PRIVATE gtest gtest_main)
  gtest_discover_tests(gx_auto_array_sizing_tests)
endif () # AURORA_ENABLE_GX

# DVD API tests
if (TARGET aurora::dvd)
  add_executable(dvd_tests test_dvd.cpp)
  target_link_libraries(dvd_tests PRIVATE aurora::dvd gtest gtest_main)

  # Pass disc image path as a compile definition if provided
  if (DEFINED DVD_TEST_IMAGE AND EXISTS "${DVD_TEST_IMAGE}")
    target_compile_definitions(dvd_tests PRIVATE DVD_TEST_IMAGE="${DVD_TEST_IMAGE}")
  endif ()

  gtest_discover_tests(dvd_tests)
endif ()

add_executable(os_alloc_tests
  os_alloc_test.cpp
  os_test_globals.cpp
  ../lib/dolphin/os/OSAlloc.cpp
  ../lib/logging.cpp
)
target_include_directories(os_alloc_tests PRIVATE
  ../include
)
target_compile_definitions(os_alloc_tests PRIVATE AURORA TARGET_PC)
target_link_libraries(os_alloc_tests PRIVATE gtest gtest_main fmt::fmt)
gtest_discover_tests(os_alloc_tests)
