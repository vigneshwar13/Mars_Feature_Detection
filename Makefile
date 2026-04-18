# Makefile for Mars Feature Detection

CC = g++
CFLAGS = `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

SRC = main.cpp helper.cpp
OBJ = $(SRC:.cpp=.o)

all: bin/Mars_Feature_Detection

bin/Mars_Feature_Detection: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $< $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) bin/Mars_Feature_Detection
