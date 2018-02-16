cmake_minimum_required(VERSION 3.5)
project(tensorflow_object_detector)
include(ExternalData)
find_package(ament_cmake REQUIRED)

##set the URL to download
set(URL "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz")
set(DOWNLOAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/data/models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz)
set(EXTRACTED_PATH ${CMAKE_CURRENT_SOURCE_DIR}/data/models)

if (NOT EXISTS "${EXTRACTED_PATH}/ssd_mobilenet_v1_coco_2017_11_17")
  if (NOT EXISTS "${DOWNLOAD_PATH}")
  message(STATUS "Downloading SSD model ....")
  file(
    DOWNLOAD "${URL}" "${DOWNLOAD_PATH}"
    SHOW_PROGRESS
  )
  endif()
  message(STATUS "Extracting...")
  execute_process(
    COMMAND cmake -E tar xzf ${DOWNLOAD_PATH}
    WORKING_DIRECTORY ${EXTRACTED_PATH}
  )
endif()

file(REMOVE ${DOWNLOAD_PATH})

#install()

ament_package()
	
#catkin_install_python(PROGRAMS scripts/detect_ros.py 
#			DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
