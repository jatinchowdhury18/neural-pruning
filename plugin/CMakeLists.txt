project(neural_pruning_plugin VERSION 1.0.0)

CPMAddPackage("gh:juce-framework/juce#8.0.5")
CPMAddPackage("gh:Chowdhury-DSP/chowdsp_utils#07f8a5b049eaa9844defdab8c13e4763d0105502")
CPMAddPackage("gh:free-audio/clap-juce-extensions#ea834dfd366ab2103b98e92eaacb581c69fc2e0b")

juce_add_plugin(neural_pruning_plugin
    COMPANY_NAME chowdsp
    PLUGIN_MANUFACTURER_CODE Chow
    PLUGIN_CODE Npp1
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Neural Pruning Plugin"
    BUNDLE_ID "org.chowdsp.neural-pruning-plugin"

    MICROPHONE_PERMISSION_ENABLED TRUE
)

clap_juce_extensions_plugin(
    TARGET neural_pruning_plugin
    CLAP_ID "org.chowdsp.neural-pruning-plugin"
    CLAP_FEATURES audio-effect
    CLAP_PROCESS_EVENTS_RESOLUTION_SAMPLES 64
    CLAP_USE_JUCE_PARAMETER_RANGES DISCRETE
)

target_compile_definitions(neural_pruning_plugin PUBLIC
    JUCE_REPORT_APP_USAGE=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_JACK=0
    JUCE_ALSA=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    CHOWDSP_BUFFER_MAX_NUM_CHANNELS=2
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../train"
)

target_link_libraries(neural_pruning_plugin
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        chowdsp::chowdsp_logging
        chowdsp::chowdsp_plugin_base
        chowdsp::chowdsp_plugin_state
        chowdsp::chowdsp_gui
        chowdsp::chowdsp_buffers
        chowdsp::chowdsp_filters
        chowdsp::chowdsp_dsp_utils
        chowdsp::chowdsp_clap_extensions
        clap_juce_extensions
        RTNeural
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
)

include(SourceFileGroup)
setup_source_group(neural_pruning_plugin PLUGIN_SRCS SOURCES
    neural_pruning_plugin.h
    neural_pruning_plugin.cpp
    lstm_model.h
    lstm_model.cpp
    plugin_editor.h
    plugin_editor.cpp
)

file(GLOB_RECURSE juce_module_sources CONFIGURE_DEPENDS
    ${juce_SOURCE_DIR}/modules/juce_*/*.cpp
    ${juce_SOURCE_DIR}/modules/juce_*/*.mm
    ${chowdsp_utils_SOURCE_DIR}/modules/*/chowdsp_*/*.cpp
)
set_source_files_properties(${juce_module_sources}
    TARGET_DIRECTORY neural_pruning_plugin
    PROPERTIES SKIP_PRECOMPILE_HEADERS TRUE SKIP_UNITY_BUILD_INCLUSION TRUE
)
set_target_properties(neural_pruning_plugin PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_MODE GROUP
    UNITY_BUILD_BATCH_SIZE 16
)

if(OSX_SIGNING_ID)
    message(STATUS "Setting up code-signing with ID: ${OSX_SIGNING_ID}")
    include(CodeSign)
    sign_plugin(neural_pruning_plugin ${OSX_SIGNING_ID})
endif()

include(Package)
package_plugin(neural_pruning_plugin)
