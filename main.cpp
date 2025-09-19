#include <iostream>
#include <fstream>
#include <deque>
#include <sycl/sycl.hpp>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"
#include "gen-cpp/calciteserver_types.h"

#include "operations/preprocessing.hpp"
#include "operations/load.hpp"
#include "operations/filter.hpp"
#include "operations/project.hpp"
#include "operations/aggregation.hpp"
#include "operations/join.hpp"
#include "operations/sort.hpp"

#include "kernels/types.hpp"

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

#define MAX_NTABLES 5

#define PERFORMANCE_MEASUREMENT_ACTIVE 1
#define PERFORMANCE_REPETITIONS 100

void print_result(const TableData<int> &table_data)
{
    int res_count = 0;
    std::cout << "Result table:" << std::endl;
    for (int i = 0; i < table_data.col_len; i++)
    {
        if (table_data.flags[i])
        {
            for (int j = 0; j < table_data.columns_size; j++) // at this point column_size should match col_number
                std::cout << ((table_data.columns[j].is_aggregate_result) ? ((unsigned long long *)table_data.columns[j].content)[i] : table_data.columns[j].content[i]) << ((j < table_data.columns_size - 1) ? " " : "");
            std::cout << "\n";
            res_count++;
        }
    }

    std::cout << "Total rows in result: " << res_count << std::endl;
}

void save_result(const TableData<int> &table_data, const std::string &data_path)
{
    std::string query_name = data_path.substr(data_path.find_last_of("/") + 1, 3);
    std::cout << "Saving result to " << query_name << ".res" << std::endl;

    std::ofstream outfile(query_name + ".res");
    if (!outfile.is_open())
    {
        std::cerr << "Could not open result file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < table_data.col_len; i++)
    {
        if (table_data.flags[i])
        {
            for (int j = 0; j < table_data.columns_size; j++) // at this point column_size should match col_number
                outfile << ((table_data.columns[j].is_aggregate_result) ? ((unsigned long long *)table_data.columns[j].content)[i] : table_data.columns[j].content[i]) << ((j < table_data.columns_size - 1) ? " " : "");
            outfile << "\n";
        }
    }

    outfile.close();
}

std::chrono::duration<double, std::milli> execute_result(const PlanResult &result, const std::string &data_path, std::ostream &perf_out = std::cout)
{
    sycl::queue queue{ sycl::gpu_selector_v };
    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    #else
    bool output_done = false;
    #endif

    TableData<int> tables[MAX_NTABLES];
    int current_table = 0,
        *output_table = sycl::malloc_host<int>(result.rels.size(), queue); // used to track the output table of each operation, in order to be referenced in the joins. other operation types just use the previous output table
    ExecutionInfo exec_info = parse_execution_info(result);
    std::vector<void *> resources; // used to track allocated resources for freeing at the end
    resources.reserve(500);        // high enough to avoid multiple reallocations
    std::map<int, std::vector<sycl::event>> dependencies; // used to track dependencies between operations

    for (const RelNode &rel : result.rels)
    {
        if (rel.relOp != RelNodeType::TABLE_SCAN)
            continue;

        #if not PERFORMANCE_MEASUREMENT_ACTIVE
        std::cout << "Table Scan on: " << rel.tables[1] << std::endl;
        #endif

        if (exec_info.loaded_columns.find(rel.tables[1]) == exec_info.loaded_columns.end())
        {
            std::cerr << "Table " << rel.tables[1] << " was never loaded." << std::endl;
            return std::chrono::duration<double, std::milli>::zero();
        }

        const std::set<int> &column_idxs = exec_info.loaded_columns[rel.tables[1]];
        tables[current_table] = loadTable(rel.tables[1], table_column_numbers[rel.tables[1]], column_idxs, resources, queue);
        tables[current_table].table_name = rel.tables[1];
        if (exec_info.group_by_columns.find(rel.tables[1]) != exec_info.group_by_columns.end())
            tables[current_table].group_by_column = exec_info.group_by_columns[rel.tables[1]];
        output_table[rel.id] = current_table;
        current_table++;
    }

    queue.wait();

    auto start = std::chrono::high_resolution_clock::now();

    for (int id : exec_info.dag_order)
    {
        const RelNode &rel = result.rels[id];
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            dependencies[rel.id] = {};
            break;
        case RelNodeType::FILTER:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto start_filter = std::chrono::high_resolution_clock::now();
            #endif

            dependencies[rel.id] = parse_filter(
                rel.condition,
                tables[output_table[rel.id - 1]],
                "",
                resources,
                queue,
                dependencies[rel.id - 1]
            );
            output_table[rel.id] = output_table[rel.id - 1];

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto end_filter = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> filter_time = end_filter - start_filter;
            std::cout << "Filter operation (" << filter_time.count() << " ms)" << std::endl;
            #endif

            break;
        }
        case RelNodeType::PROJECT:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto start_project = std::chrono::high_resolution_clock::now();
            #endif

            dependencies[rel.id] = parse_project(
                rel.exprs,
                tables[output_table[rel.id - 1]],
                resources,
                queue,
                dependencies[rel.id - 1]
            );

            output_table[rel.id] = output_table[rel.id - 1];

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto end_project = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> project_time = end_project - start_project;
            std::cout << "Project operation (" << project_time.count() << " ms)" << std::endl;
            #endif

            break;
        }
        case RelNodeType::AGGREGATE:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto start_aggregate = std::chrono::high_resolution_clock::now();
            #endif

            dependencies[rel.id] = parse_aggregate(
                tables[output_table[rel.id - 1]],
                rel.aggs[0],
                rel.group,
                resources,
                queue,
                dependencies[rel.id - 1]
            );

            output_table[rel.id] = output_table[rel.id - 1];

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto end_aggregate = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> aggregate_time = end_aggregate - start_aggregate;
            std::cout << "Aggregate operation (" << aggregate_time.count() << " ms)" << std::endl;
            #endif

            break;
        }
        case RelNodeType::JOIN:
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto start_join = std::chrono::high_resolution_clock::now();
            #endif

            std::vector<sycl::event> join_dependencies;
            join_dependencies.insert(join_dependencies.end(), dependencies[rel.inputs[0]].begin(), dependencies[rel.inputs[0]].end());
            join_dependencies.insert(join_dependencies.end(), dependencies[rel.inputs[1]].begin(), dependencies[rel.inputs[1]].end());

            dependencies[rel.id] = parse_join(
                rel,
                tables[output_table[rel.inputs[0]]],
                tables[output_table[rel.inputs[1]]],
                exec_info.table_last_used,
                resources,
                queue,
                join_dependencies
            );

            output_table[rel.id] = output_table[rel.inputs[0]];

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto end_join = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> join_time = end_join - start_join;
            std::cout << "Join operation (" << join_time.count() << " ms)" << std::endl;
            #endif

            break;
        }
        case RelNodeType::SORT:
        {
            queue.wait();
            auto start_sort = std::chrono::high_resolution_clock::now();

            parse_sort(rel, tables[output_table[rel.id - 1]], queue);
            output_table[rel.id] = output_table[rel.id - 1];

            dependencies[rel.id] = {};

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto end_sort = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> sort_time = end_sort - start_sort;
            std::chrono::duration<double, std::milli> exec_no_sort = start_sort - start;
            std::cout << "Execution time without sort: " << exec_no_sort.count() << " ms\n"
                << "Sort operation (" << sort_time.count() << " ms)" << std::endl;
            #else
            std::chrono::duration<double, std::milli> exec_no_sort = start_sort - start;
            perf_out << exec_no_sort.count() << '\n';
            output_done = true;
            #endif

            break;
        }
        default:
            std::cout << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            break;
        }
    }

    auto end_before_wait = std::chrono::high_resolution_clock::now();

    queue.wait();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> time_before_wait = end_before_wait - start;
    std::chrono::duration<double, std::milli> exec_time = end - start;

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Execution time: " << exec_time.count() << " ms - " << time_before_wait.count() << " ms (before wait)" << std::endl;
    #else
    if (!output_done)
        perf_out << exec_time.count() << '\n';
    #endif

    TableData<int> &final_table = tables[output_table[result.rels.size() - 1]];
    for (int i = 0; i < final_table.columns_size; i++)
    {
        if (final_table.columns[i].has_ownership)
        {
            if (final_table.columns[i].is_aggregate_result)
            {
                uint64_t *host_col = sycl::malloc_host<uint64_t>(final_table.col_len, queue);
                queue.copy((uint64_t *)final_table.columns[i].content, host_col, final_table.col_len).wait();
                sycl::free(final_table.columns[i].content, queue);
                final_table.columns[i].content = (int *)host_col;
            }
            else
            {
                int *host_col = sycl::malloc_host<int>(final_table.col_len, queue);
                queue.copy(final_table.columns[i].content, host_col, final_table.col_len).wait();
                sycl::free(final_table.columns[i].content, queue);
                final_table.columns[i].content = host_col;
            }
        }
        else
            std::cout << "!!!!!!!!!! Column " << i << " does not have ownership, skipping copy to host !!!!!!!!!!" << std::endl;
    }

    bool *host_flags = sycl::malloc_host<bool>(final_table.col_len, queue);
    queue.copy(final_table.flags, host_flags, final_table.col_len).wait();
    sycl::free(final_table.flags, queue);
    final_table.flags = host_flags;

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    // print_result(final_table);
    save_result(final_table, data_path);

    start = std::chrono::high_resolution_clock::now();
    #endif

    for (int i = 0; i < current_table; i++)
    {
        sycl::free(tables[i].flags, queue);
        for (int j = 0; j < tables[i].columns_size; j++)
            if (tables[i].columns[j].has_ownership)
                sycl::free(tables[i].columns[j].content, queue);
        sycl::free(tables[i].columns, queue);
    }
    sycl::free(output_table, queue);

    for (void *res : resources)
        sycl::free(res, queue);

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> free_time = end - start;
    std::cout << "Free resources time: " << free_time.count() << " ms" << std::endl;
    #endif

    return exec_time;
}

int main(int argc, char **argv)
{
    std::shared_ptr<TTransport> socket(new TSocket("localhost", 5555));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    CalciteServerClient client(protocol);
    std::string sql;

    if (argc == 2)
    {
        std::ifstream file(argv[1]);
        if (!file.is_open())
        {
            std::cerr << "Could not open file: " << argv[1] << std::endl;
            return 1;
        }

        sql.assign((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());

        file.close();
    }
    else
    {
        sql = "select sum(lo_revenue)\
        from lineorder, ddate, part, supplier\
        where lo_orderdate = d_datekey\
        and lo_partkey = p_partkey\
        and lo_suppkey = s_suppkey;";
    }

    try
    {
        // std::cout << "SQL Query: " << sql << std::endl;
        transport->open();
        std::cout << "Transport opened successfully." << std::endl;

        #if PERFORMANCE_MEASUREMENT_ACTIVE
        std::string sql_filename = argv[1];
        std::string query_name = sql_filename.substr(sql_filename.find_last_of("/") + 1, 3);
        std::ofstream perf_file(query_name + "-performance.log", std::ios::out | std::ios::trunc);
        if (!perf_file.is_open())
        {
            std::cerr << "Could not open performance log file: " << query_name << "-performance.log" << std::endl;
            return 1;
        }

        for (int i = 0; i < PERFORMANCE_REPETITIONS; i++)
        {
            PlanResult result;

            auto start = std::chrono::high_resolution_clock::now();
            client.parse(result, sql);
            auto exec_time = execute_result(result, argv[1], perf_file);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> total_time = end - start;

            std::cout << "Repetition " << i + 1 << "/" << PERFORMANCE_REPETITIONS
                << " - " << exec_time.count() << " ms - "
                << total_time.count() << " ms" << std::endl;
        }
        perf_file.close();
        #else
        PlanResult result;
        client.parse(result, sql);

        // std::cout << "Result: " << result << std::endl;

        execute_result(result, argv[1]);
        #endif

        // client.shutdown();

        transport->close();
    }
    catch (TTransportException &e)
    {
        std::cerr << "Transport exception: " << e.what() << std::endl;
    }
    catch (TException &e)
    {
        std::cerr << "Thrift exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}