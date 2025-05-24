function(sign_plugin plugin_target signing_id)
    foreach(target IN ITEMS ${plugin_target}_VST3 ${plugin_target}_AU ${plugin_target}_CLAP)
        if(NOT TARGET ${target})
            continue()
        endif()

        get_target_property(plugin_artefact_file_path ${target} JUCE_PLUGIN_ARTEFACT_FILE)
        if(APPLE)
            add_custom_command(TARGET ${target}
                POST_BUILD
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMAND ${CMAKE_COMMAND} -E echo "Signing target ${target} at location ${plugin_artefact_file_path}"
                VERBATIM
                COMMAND codesign --sign "${signing_id}" --deep --strict --options=runtime --timestamp --force --verbose "${plugin_artefact_file_path}"
                VERBATIM
                COMMAND codesign --verify --verbose "${plugin_artefact_file_path}"
            )
        endif()
    endforeach()
endfunction()
