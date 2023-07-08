compile:
	zig cc \
		-Wall -Wextra -Wpedantic -Wshadow -Wcast-qual -Wstrict-prototypes \
		-Wpointer-arith -Wdouble-promotion -Wno-unused-function -Werror=vla \
		-mavx -mavx2 -mfma -mf16c -msse3 -O3 -DNDEBUG -std=gnu11 \
		-o ggml.o -c ggml.c
	zig c++ \
		-I/usr/lib/gcc/x86_64-linux-gnu/12/include -Wall -fpic -c -o bert.o bert.cpp
	zig cc \
		-I/usr/lib/gcc/x86_64-linux-gnu/12/include \
		-I/home/jjzazuet/Applications/zulu17.42.21-ca-crac-jdk17.0.7-linux_x64/include \
		-I/home/jjzazuet/Applications/zulu17.42.21-ca-crac-jdk17.0.7-linux_x64/include/linux \
		-std=gnu11 -Wall -c -o bert-jni.o bert-jni.c
	zig cc ggml.o bert.o bert-jni.o -shared -lm -lc -lstdc++ -o libbert-jni.so

clean:
	rm -rfv ggml.o bert.o bert-jni.o libbert-jni.so
