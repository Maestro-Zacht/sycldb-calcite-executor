CXX := clang++ -fsycl -fsycl-embed-ir -Wall -fsycl-targets=nvptx64-nvidia-cuda,spir64
CXXFLAGS := -std=c++20 -O3 -Ikernels -Igen-cpp -Ioperations -Imodels -I/usr/local/include
LDFLAGS := -L/usr/local/lib -lthrift -Wl,-rpath=/usr/local/lib

SRC := main.cpp gen-cpp/CalciteServer.cpp gen-cpp/calciteserver_types.cpp
HEADERS := gen-cpp/CalciteServer.h gen-cpp/calciteserver_types.h $(wildcard kernels/*.hpp) $(wildcard operations/*.hpp) $(wildcard models/*.hpp)
TARGET := client

QUERY_NAMES := $(patsubst %.sql, %, $(notdir $(wildcard ./queries/transformed/q*.sql)))
RESULT_FILES = $(notdir $(wildcard ./q*.res))
RESULT_NAMES = $(patsubst %.res, %, $(RESULT_FILES))

.PHONY: clean check fullcheck q%


$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

q%: q%.result
	diff ./reference_results/$@.txt ./$@.res

q%.result: $(TARGET)
	./$(TARGET) ./queries/transformed/q$*.sql

clean:
	-rm client
	-rm q*.res

check:
	@for q in $(RESULT_NAMES); do \
		diff -q ./reference_results/$$q.txt ./$$q.res; \
		echo "checked $$q"; \
	done

fullcheck: $(QUERY_NAMES) $(RESULT_NAMES)