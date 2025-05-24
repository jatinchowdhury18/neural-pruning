function(package_plugin plugin_target)
    set(package_target "${plugin_target}_PACKAGE")
    add_custom_target(${package_target}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E echo "Making package directory"
        COMMAND ${CMAKE_COMMAND} -E remove_directory package
        COMMAND ${CMAKE_COMMAND} -E make_directory package
    )

    foreach(target IN ITEMS ${plugin_target}_VST3 ${plugin_target}_AU ${plugin_target}_CLAP)
        if(NOT TARGET ${target})
            continue()
        endif()

        message(STATUS "Adding target ${target} to package")
        add_dependencies(${package_target} ${target})
        get_target_property(plugin_artefact_file_path ${target} JUCE_PLUGIN_ARTEFACT_FILE)
        cmake_path(GET plugin_artefact_file_path FILENAME plugin_file_name)
        add_custom_command(TARGET ${package_target}
            POST_BUILD
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMAND ${CMAKE_COMMAND} -E echo "Copying target ${plugin_artefact_file_path} to package/${plugin_file_name}"
            COMMAND ${CMAKE_COMMAND} "-Dsrc=${plugin_artefact_file_path}" "-Ddest=package" "-P" "${JUCE_CMAKE_UTILS_DIR}/copyDir.cmake"
        )
    endforeach()
    add_custom_command(TARGET ${package_target}
        POST_BUILD
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E echo "Zipping package directory"
        COMMAND ${CMAKE_COMMAND} -E tar "cf" "${plugin_target}-${CMAKE_SYSTEM_NAME}-${PROJECT_VERSION}.zip" --format=zip -- package
    )
endfunction()
