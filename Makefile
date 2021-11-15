PLATFORM=max
include $(cdir)Makefile.in.$(PLATFORM)

# Trial path flag so the made files don't get mixed up
P?=./

DRIVERS=test MPI_timing

OBJECTS=test.o matrix.o tensor.o tt.o readArray_MPI.o

mkfile_path=$(abspath $(lastword $(MAKEFILE_LIST)))
cdir?=$(dir $(mkfile_path))

THREADS?=1

MPICOMMANDS=-np $(THREADS) -oversubscribe

D?=3
N?=500 500 500
R?=1 25 25 1
NPS?=4 1 4
M?=400
GAMMA?=10


SOLVETYPE=3
TENSORTYPE=1

MID=-1

TESTPATH=./timing_test/

# ===
# Main driver and sample run

.ONESHELL:
all:
	cp Makefile $PMakefileCopy
	cd $(P); pwd;
	make compile -f MakefileCopy cdir=$(cdir)


.PHONY: compile
compile:	$(DRIVERS)

MPI_timing: MPI_timing.o matrix.o tensor.o sketch.o tt.o readArray_MPI.o PSTT.o VTime.o SSTT.o my_io.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(LIBBLAS)

MPI_timing.o: $(cdir)MPI_timing.c $(cdir)matrix.h $(cdir)tensor.h $(cdir)tt.h $(cdir)readArray_MPI.h $(cdir)sketch.h $(cdir)PSTT.h $(cdir)VTime.h $(cdir)SSTT.h $(cdir)my_io.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

my_io.o: $(cdir)my_io.c $(cdir)tt.h $(cdir)VTime.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

test: test.o matrix.o tensor.o sketch.o tt.o readArray_MPI.o PSTT.o VTime.o SSTT.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(LIBBLAS)

test.o: $(cdir)test.c $(cdir)matrix.h $(cdir)tensor.h $(cdir)tt.h $(cdir)readArray_MPI.h $(cdir)sketch.h $(cdir)PSTT.h $(cdir)VTime.h $(cdir)SSTT.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

SSTT.o: $(cdir)SSTT.c $(cdir)matrix.h $(cdir)tensor.h $(cdir)tt.h $(cdir)sketch.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

PSTT.o: $(cdir)PSTT.c $(cdir)sketch.h $(cdir)matrix.h $(cdir)tensor.h $(cdir)tt.h $(cdir)VTime.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

readArray_MPI.o: $(cdir)readArray_MPI.c $(cdir)matrix.h $(cdir)tensor.h $(cdir)tt.h $(cdir)sketch.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

tt.o: $(cdir)tt.c $(cdir)matrix.h $(cdir)tensor.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

sketch.o: $(cdir)sketch.c $(cdir)matrix.h $(cdir)tensor.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

tensor.o: $(cdir)tensor.c $(cdir)matrix.h
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

matrix.o: $(cdir)matrix.c
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

VTime.o: $(cdir)VTime.c
	$(CC) $(CFLAGS) -c $(INCBLAS) $<

# ===
# Run the compiled code
.PHONY: time
time:
	make
	mpirun $(MPICOMMANDS) ./MPI_timing $(SOLVETYPE) $(TENSORTYPE) -d $(D) -n $(N) -r $(R) -p $(TESTPATH) -e $(NPS) -m $(MID) -h 0 -M $(M) -g $(GAMMA)

IGNORE=--ignore="(MPI_Recv)|(PMPI_Gather)|(PMPI_Send)"
.PHONY: profile
profile:
	make
	rm -rf $(TESTPATH).*.heap $(TESTPATH)*.txt $(TESTPATH)*.pdf
	mpirun $(MPICOMMANDS) ./MPI_timing $(SOLVETYPE) $(TENSORTYPE) -d $(D) -n $(N) -r $(R) -p $(TESTPATH) -e $(NPS) -m $(MID) -h 1 -M $(M) -g $(GAMMA)
	#google-pprof --pdf --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).tt0.0001.heap > $(TESTPATH)tt0.pdf
	#google-pprof --text --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).tt0.0001.heap > $(TESTPATH)tt0.txt
	#google-pprof --pdf --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).ten0.0002.heap > $(TESTPATH)ten0.pdf
	#google-pprof --text --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).ten0.0002.heap > $(TESTPATH)ten0.txt
	google-pprof --pdf --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).SSTT0.0003.heap > $(TESTPATH)SSTT0.pdf
	google-pprof --text --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).SSTT0.0003.heap > $(TESTPATH)SSTT0.txt
	google-pprof --pdf --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).PSTT0.0003.heap > $(TESTPATH)PSTT0.pdf
	google-pprof --text --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).PSTT0.0003.heap > $(TESTPATH)PSTT0.txt
	google-pprof --pdf --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).PSTT_onepass0.0003.heap > $(TESTPATH)PSTT_onepass0.pdf
	google-pprof --text --alloc_space $(IGNORE) $(P)MPI_timing $(TESTPATH).PSTT_onepass0.0003.heap > $(TESTPATH)PSTT_onepass0.txt





.PHONY: run
run:
	make
	mpirun $(MPICOMMANDS) ./test

.PHONY: profile2
profile2:
	make
	export HEAPPROFILE=$(TESTPATH).0 
	echo $$HEAPPROFILE
	./MPI_timing $(SOLVETYPE) $(TENSORTYPE) -d $(D) -n $(N) -r $(R) -p $(TESTPATH) -e $(NPS) -m $(MID) -h 0 -M $(M) -g $(GAMMA)
	google-pprof --pdf --alloc_space $(P)test $(TESTPATH).0.0001.heap > $(TESTPATH)0.pdf
	google-pprof --text --alloc_space $(P)test $(TESTPATH).0.0001.heap > $(TESTPATH)0.txt



.PHONY: grind
grind:
	make
	mpirun $(MPICOMMANDS) valgrind --leak-check=full ./MPI_timing $(SOLVETYPE) $(TENSORTYPE) -d $(D) -n $(N) -r $(R) -p $(TESTPATH) -e $(NPS) -m $(MID) -h 0 -M $(M) -g $(GAMMA)
	#mpirun $(MPICOMMANDS) valgrind --leak-check=full ./test

.PHONY: gdb
gdb:
	make
	gdb mpirun $(MPICOMMANDS) ./test

# ===
# Clean up
.ONESHELL:
clean:
	cd $(P); pwd;
	rm -f $(DRIVERS) *.o MakefileCopy heapprof.*
