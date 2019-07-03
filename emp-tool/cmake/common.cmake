if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # use rpath on macOS
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(${CMAKE_SOURCE_DIR})

## Build type
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_SOURCE_DIR}/cmake)

#Compilation flags
set (CMAKE_C_FLAGS "-pthread -Wall -march=native -O3 -maes -mrdseed -std=c++0x")
set (CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

#Testing macro
macro (add_test_with_lib _name libs)
	add_executable(${_name} "test/${_name}.cpp")
	target_link_libraries(${_name}  ${RELIC_LIBRARIES} ${OPENSSL_LIBRARIES} ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${libs}) 
endmacro()


OPTION(NETIO_USE_TLS "Turning on TLS for NetIO" OFF)
OPTION(NETIO_USE_TLS_NONAMECHECK "Skipping certificate common name/IP matching check in NetIO TLS" OFF)

set(NETIO_CA_CERTIFICATE "" CACHE STRING "Alternative CA certificate file location")
set(NETIO_MY_CERTIFICATE "" CACHE STRING "Alternative this party's certificate file location")
set(NETIO_MY_PRIVATE_KEY "" CACHE STRING "ALternative this party's private key file location")

IF(${NETIO_USE_TLS})
	add_definitions(-DNETIO_USE_TLS)
	message("${Green}-- TLS: ON${ColourReset}")

	IF(${NETIO_USE_TLS_NONAMECHECK})
		add_definitions(-DNETIO_USE_TLS_NONAMECHECK)
		message("${Red}-- TLS insecure: No certificate name check${ColourReset}")
	ENDIF(${NETIO_USE_TLS_NONAMECHECK})

	IF(NOT "${NETIO_CA_CERTIFICATE}" STREQUAL "")
		add_definitions(-DNETIO_CA_CERTIFICATE=\"${NETIO_CA_CERTIFICATE}\")
		message("${Green}-- TLS CA certificate: ${NETIO_CA_CERTIFICATE}${ColourReset}")
	ELSE(NOT "${NETIO_CA_CERTIFICATE}" STREQUAL "")
		message("${Green}-- TLS CA certificate: ./certificates/ca.pem (default}${ColourReset}")
	ENDIF(NOT "${NETIO_CA_CERTIFICATE}" STREQUAL "")

	IF(NOT "${NETIO_MY_CERTIFICATE}" STREQUAL "")
		add_definitions(-DNETIO_MY_CERTIFICATE=\"${NETIO_MY_CERTIFICATE}\")
		message("${Green}-- TLS my certificate: ${NETIO_MY_CERTIFICATE}${ColourReset}")
	ELSE(NOT "${NETIO_MY_CERTIFICATE}" STREQUAL "")
		message("${Green}-- TLS my certificate: ./certificates/my_certificate.pem (default)${ColourReset}")
	ENDIF(NOT "${NETIO_MY_CERTIFICATE}" STREQUAL "")

	IF(NOT "${NETIO_MY_PRIVATE_KEY}" STREQUAL "")
		add_definitions(-DNETIO_MY_PRIVATE_KEY=\"${NETIO_MY_PRIVATE_KEY}\")
		message("${Green}-- TLS my private key: ${NETIO_MY_PRIVATE_KEY}${ColourReset}")
	ELSE(NOT "${NETIO_MY_PRIVATE_KEY}" STREQUAL "")
		message("${Green}-- TLS my private key: ./certificates/my_private_key.key (default)${ColourReset}")
	ENDIF(NOT "${NETIO_MY_PRIVATE_KEY}" STREQUAL "")
ELSE(${NETIO_USE_TLS})
	message("${Red}-- TLS: OFF${ColourReset}")
ENDIF(${NETIO_USE_TLS})
