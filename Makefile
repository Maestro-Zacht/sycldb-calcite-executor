CXX := clang++ -fsycl -Wall -fsycl-targets=nvptx64-nvidia-cuda
CXXFLAGS := -std=c++20 -O3 -Ikernels -Igen-cpp -I/usr/local/include
LDFLAGS := -L/usr/local/lib -lthrift -Wl,-rpath=/usr/local/lib

SRC := main.cpp gen-cpp/CalciteServer.cpp gen-cpp/calciteserver_types.cpp
HEADERS := gen-cpp/CalciteServer.h gen-cpp/calciteserver_types.h $(wildcard kernels/*.hpp)
TARGET := client

QUERY_NAMES := $(patsubst %.sql, %, $(notdir $(wildcard ./queries/transformed/q*.sql)))
RESULT_FILES = $(notdir $(wildcard ./q*.res))
RESULT_NAMES = $(patsubst %.res, %, $(RESULT_FILES))

.PHONY: clean check fullcheck q%


$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

q%: $(TARGET)
	./$(TARGET) ./queries/transformed/$@.sql

q%.res: q%
	diff ./reference_results/$<.txt ./$@

clean:
	-rm client
	-rm q*.res

check:
	@echo "========== Checking results... =========="
	@for q in $(RESULT_NAMES); do \
		diff -q ./reference_results/$$q.txt ./$$q.res; \
		echo "checked $$q"; \
	done

fullcheck: $(QUERY_NAMES) $(RESULT_FILES)