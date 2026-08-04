cmake_minimum_required(VERSION 3.16)
project(ctkm_extractor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CTKM_BUILD_TESTS "Build unit tests (cần tải Catch2 qua FetchContent)" ON)
option(CTKM_WARNINGS_AS_ERRORS "Coi warning là lỗi" OFF)

# --------------------------------------------------------------------------
# Dependency bắt buộc: OpenCV
# --------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
include(GNUInstallDirs)

# --------------------------------------------------------------------------
# Dependency tuỳ chọn 1: ONNXRuntime (provider mặc định paddle_vietocr)
#
# Chỉ định thủ công:  cmake -B build -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime
# Thiếu thì vẫn build được, provider mặc định sẽ báo không khả dụng và tự
# fallback sang Tesseract - đúng hành vi try/except ImportError của bản Python.
# --------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Thư mục ONNXRuntime prebuilt (chứa include/ và lib/)")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT_DIR}/include /opt/onnxruntime/include /usr/local/include
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT_DIR}/lib /opt/onnxruntime/lib /usr/local/lib)

# --------------------------------------------------------------------------
# Dependency tuỳ chọn 2: Tesseract + Leptonica (provider fallback)
# --------------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(TESSERACT QUIET tesseract)
  pkg_check_modules(LEPTONICA QUIET lept)
endif()
if(NOT TESSERACT_FOUND)
  find_path(TESSERACT_INCLUDE_DIR NAMES tesseract/baseapi.h)
  find_library(TESSERACT_LIBRARY NAMES tesseract)
  find_path(LEPTONICA_INCLUDE_DIR NAMES leptonica/allheaders.h)
  find_library(LEPTONICA_LIBRARY NAMES lept leptonica)
  if(TESSERACT_INCLUDE_DIR AND TESSERACT_LIBRARY AND LEPTONICA_LIBRARY)
    set(TESSERACT_FOUND TRUE)
    set(TESSERACT_INCLUDE_DIRS ${TESSERACT_INCLUDE_DIR} ${LEPTONICA_INCLUDE_DIR})
    set(TESSERACT_LIBRARIES ${TESSERACT_LIBRARY} ${LEPTONICA_LIBRARY})
  endif()
endif()

# --------------------------------------------------------------------------
# Thư viện lõi (dùng chung cho binary và test)
# --------------------------------------------------------------------------
add_library(ctkm_core STATIC
  src/util/Utf8.cpp
  src/util/Log.cpp
  src/util/Json.cpp
  src/ocr/OCRProvider.cpp
  src/ocr/Preprocess.cpp
  src/ocr/OnnxDetector.cpp
  src/ocr/OnnxVietOCR.cpp
  src/ocr/PaddleVietOcrProvider.cpp
  src/ocr/TesseractOCR.cpp
  src/ocr/ProviderFactory.cpp
  src/table/Reconstruct.cpp
  src/table/Morphology.cpp
  src/table/PPStructure.cpp
  src/table/TableBuilder.cpp
  src/extraction/Schema.cpp
  src/extraction/FieldParsers.cpp
  src/extraction/Extractor.cpp)

target_include_directories(ctkm_core PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ctkm_core PUBLIC ${OpenCV_LIBS})

# Đường dẫn schema mặc định khi không truyền --schema và không có ./schema.json.
target_compile_definitions(ctkm_core
  PUBLIC CTKM_DEFAULT_SCHEMA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/schema.json"
  # Vị trí schema so với thư mục chứa binary sau khi `cmake --install`, nhờ đó
  # bản đã cài chạy được từ thư mục bất kỳ mà không cần đặt CTKM_SCHEMA.
  PUBLIC CTKM_INSTALL_SCHEMA_RELATIVE_PATH="../${CMAKE_INSTALL_DATADIR}/ctkm_extractor/schema.json")

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  message(STATUS "ONNXRuntime: ${ONNXRUNTIME_LIBRARY}")
  target_compile_definitions(ctkm_core PUBLIC CTKM_WITH_ONNXRUNTIME)
  target_include_directories(ctkm_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(ctkm_core PUBLIC ${ONNXRUNTIME_LIBRARY})
else()
  message(WARNING
    "Không tìm thấy ONNXRuntime - provider 'paddle_vietocr' sẽ không khả dụng. "
    "Cấu hình lại với -DONNXRUNTIME_ROOT_DIR=<đường dẫn> để bật.")
endif()

if(TESSERACT_FOUND)
  message(STATUS "Tesseract: ${TESSERACT_LIBRARIES}")
  target_compile_definitions(ctkm_core PUBLIC CTKM_WITH_TESSERACT)
  target_include_directories(ctkm_core PUBLIC ${TESSERACT_INCLUDE_DIRS} ${LEPTONICA_INCLUDE_DIRS})
  target_link_directories(ctkm_core PUBLIC ${TESSERACT_LIBRARY_DIRS} ${LEPTONICA_LIBRARY_DIRS})
  # Leptonica phải được link tường minh: pkg-config của tesseract không kéo theo
  # lept, dẫn tới lỗi "DSO missing from command line" khi link.
  target_link_libraries(ctkm_core PUBLIC ${TESSERACT_LIBRARIES} ${LEPTONICA_LIBRARIES})
else()
  message(WARNING
    "Không tìm thấy Tesseract - provider fallback sẽ không khả dụng. "
    "Cài bằng: sudo apt install libtesseract-dev libleptonica-dev tesseract-ocr-vie")
endif()

target_compile_options(ctkm_core PRIVATE -Wall -Wextra)
if(CTKM_WARNINGS_AS_ERRORS)
  target_compile_options(ctkm_core PRIVATE -Werror)
endif()

# --------------------------------------------------------------------------
# Binary
# --------------------------------------------------------------------------
add_executable(ctkm_extractor src/main.cpp)
target_link_libraries(ctkm_extractor PRIVATE ctkm_core)

# Copy schema.json cạnh binary để chạy trực tiếp trong thư mục build.
add_custom_command(TARGET ctkm_extractor POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/schema.json $<TARGET_FILE_DIR:ctkm_extractor>/schema.json)

# --------------------------------------------------------------------------
# Cài đặt: cmake --install build --prefix <đích>
#
# schema.json được cài vào <prefix>/share/ctkm_extractor/ và binary tìm thấy nó
# qua biến môi trường CTKM_SCHEMA hoặc cờ --schema.
# --------------------------------------------------------------------------
# Giữ đường dẫn tới thư viện đã link (nhất là libonnxruntime.so nằm ngoài các thư
# mục hệ thống) để binary sau khi cài chạy được mà không cần LD_LIBRARY_PATH.
set_target_properties(ctkm_extractor PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
  BUILD_WITH_INSTALL_RPATH FALSE)
install(TARGETS ctkm_extractor RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/schema.json
        DESTINATION ${CMAKE_INSTALL_DATADIR}/ctkm_extractor)

# --------------------------------------------------------------------------
# Test (Catch2 v3 qua FetchContent)
# --------------------------------------------------------------------------
if(CTKM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)

  add_executable(ctkm_tests
    tests/test_field_parsers.cpp
    tests/test_extractor.cpp
    tests/test_morphology.cpp
    tests/test_tesseract_psm.cpp)
  target_link_libraries(ctkm_tests PRIVATE ctkm_core Catch2::Catch2WithMain)
  target_compile_definitions(ctkm_tests
    PRIVATE CTKM_TEST_SCHEMA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/schema.json")

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(CTest)
  include(Catch)
  catch_discover_tests(ctkm_tests)
endif()
