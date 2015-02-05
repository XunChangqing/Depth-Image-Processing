set(APPLICATION_NAMES
    face_detection
    )

foreach(APPLICATION_NAME ${APPLICATION_NAMES})
  add_subdirectory(${APPLICATION_NAME})
endforeach(APPLICATION_NAME)
