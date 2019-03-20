add_library(llmpx MODULE llmpx.cpp)

target_compile_features(llmpx PRIVATE cxx_range_for cxx_auto_type)

set_target_properties(llmpx PROPERTIES
	COMPILE_FLAGS "-fno-rtti"
	)


