compile:
	$(CC) \
		-I/usr/lib/gcc/x86_64-linux-gnu/10/include \
		-I/home/jjzazuet/Applications/zulu11.52.13-ca-jdk11.0.13-linux_x64/include \
		-I/home/jjzazuet/Applications/zulu11.52.13-ca-jdk11.0.13-linux_x64/include/linux \
		-Wall -c -o rwkv_context.o rwkv_context.c
	$(CC) -L. -lrwkv -shared -o librwkv_context.so rwkv_context.o

clean:
	rm -rfv rwkv_context.o librwkv_context.so
