# Makefile for C Compiler Lexer

CC = gcc
CFLAGS = -Wall -Wextra -g
TARGET = lexer

all: $(TARGET)

$(TARGET): lexer.c
	$(CC) $(CFLAGS) -o $(TARGET) lexer.c

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	@echo "Running lexer tests..."
	./$(TARGET)

.PHONY: all clean run test
