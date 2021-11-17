include ../Makefile.inc

LIB=../lib/libparalleltt.a
OBJ=matrix.o tensor.o tt.o sketch.o PSTT.o SSTT.o VTime.o

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(LIB): $(OBJ)
	$(AR) $(ARFLAGS) $@ $(OBJ)

.PHONY: clean
clean:
	rm *.o $(LIB)