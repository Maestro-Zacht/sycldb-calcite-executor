CXX := clang++ -fsycl -Wall -fsycl-targets=nvptx64-nvidia-cuda
CXXFLAGS := -std=c++20 -O3 -Ikernels -Igen-cpp -I/usr/local/include
LDFLAGS := -L/usr/local/lib -lthrift -Wl,-rpath=/usr/local/lib

SRC := main.cpp gen-cpp/CalciteServer.cpp gen-cpp/calciteserver_types.cpp
TARGET := client

QUERIES := $(wildcard ./queries/transformed/q*.sql)
QUERY_NAMES := $(patsubst %.sql, %, $(notdir $(QUERIES)))
RESULT_FILES = $(patsubst %.res, %, $(notdir $(wildcard ./q*.res)))

.PHONY: clean check fullcheck q%


$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

q%: $(TARGET)
	./$(TARGET) ./queries/transformed/$@.sql

clean:
	-rm client
	-rm q*.res

check:
	@echo "========== Checking results... =========="
	@for q in $(RESULT_FILES); do \
		diff -q ./reference_result/$$q.txt ./$$q.res; \
		echo "checked $$q"; \
	done

fullcheck: $(QUERY_NAMES) check