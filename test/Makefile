include ../Makefile.inc

OBJ=readArray_MPI.o
MPICOMMANDS=-np $(THREADS) -oversubscribe
LIBPTT=-lparalleltt

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

example: example.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -L../lib/ $(LIBPTT) $(LIBS) $(LIBBLAS)

.PHONY: test
test:
	make
	mpirun -np $(THREADS) example

.PHONY: clean
clean:
	rm *.o example