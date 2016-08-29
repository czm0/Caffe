#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gflags-shared" for configuration "Debug"
set_property(TARGET gflags-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gflags-shared PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflagsd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflagsd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS gflags-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_gflags-shared "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflagsd.lib" "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflagsd.dll" )

# Import target "gflags_nothreads-shared" for configuration "Debug"
set_property(TARGET gflags_nothreads-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gflags_nothreads-shared PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflags_nothreadsd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflags_nothreadsd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS gflags_nothreads-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_gflags_nothreads-shared "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflags_nothreadsd.lib" "${_IMPORT_PREFIX}/x64/v120/dynamic/Lib/gflags_nothreadsd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
