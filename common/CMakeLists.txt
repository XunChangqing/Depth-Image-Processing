set(MODULE_NAME common)

SET(MODULE_DEPENDENCIES
    )

include(includes.cmake)

foreach(DEPENDENT ${MODULE_DEPENDENCIES})
  include(${PROJECT_SOURCE_DIR}/${DEPENDENT}/includes.cmake)

  set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/${DEPENDENT}/include
    )
endforeach(DEPENDENT)

set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    include/
    )

include_directories(${INCLUDE_DIRS})

set(INCS
    include/dip/${MODULE_NAME}/distance.h
    include/dip/${MODULE_NAME}/macros.h
    include/dip/${MODULE_NAME}/types.h
    )

set(SRCS
    src/distance.cpp
    )

set(LIBS
    ${MODULE_DEPENDENCIES}
    )

add_library(${MODULE_NAME} ${SRCS} ${INCS})
target_link_libraries(${MODULE_NAME} ${LIBS})

install(TARGETS ${MODULE_NAME} DESTINATION ${LIB_INSTALL_DIR}/Release
        CONFIGURATIONS Release)
install(TARGETS ${MODULE_NAME} DESTINATION ${LIB_INSTALL_DIR}/Debug
        CONFIGURATIONS Debug)
install(FILES ${INCS} DESTINATION ${INCLUDE_INSTALL_DIR}/${MODULE_NAME})

