add_library(aurora_audio STATIC
        lib/audio.cpp
)
add_library(aurora::audio ALIAS aurora_audio)
set_target_properties(aurora_audio PROPERTIES FOLDER "aurora")

target_link_libraries(aurora_audio PUBLIC aurora::core)
