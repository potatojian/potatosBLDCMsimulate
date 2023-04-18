MAKER = gcc
CFLAGS = -std=gnu99 -lm -lpthread -ldnnl

run: Main.c BLDCMotorModel.o RLModel.o
	$(MAKER) $(CFLAGS) Main.c BLDCMotorModel.o RLModel.o -o run

BLDCMotorModel.o: BLDCMotorModel.c
	$(MAKER) $(CFLAGS) -c BLDCMotorModel.c

GeneticAlgorithm.o: RLModel.c
	$(MAKER) $(CFLAGS) -c RLModel.c

clean:
	rm *.o *.csv *.tmp *.svg run