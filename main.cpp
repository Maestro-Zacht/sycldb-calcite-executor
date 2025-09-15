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

void execute_result(const PlanResult &result, const std::string &data_path)
{
    sycl::queue queue{sycl::gpu_selector_v};
    std::cout << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    TableData<int> tables[MAX_NTABLES];
    int current_table = 0,
        *output_table = sycl::malloc_shared<int>(result.rels.size(), queue); // used to track the output table of each operation, in order to be referenced in the joins. other operation types just use the previous output table
    ExecutionInfo exec_info = parse_execution_info(result);
    std::vector<void *> resources; // used to track allocated resources for freeing at the end
    resources.reserve(100);        // high enough to avoid multiple reallocations

    for (const RelNode &rel : result.rels)
    {
        if (rel.relOp != RelNodeType::TABLE_SCAN)
            continue;
        std::cout << "Table Scan on: " << rel.tables[1] << std::endl;
        if (exec_info.loaded_columns.find(rel.tables[1]) == exec_info.loaded_columns.end())
        {
            std::cout << "Table " << rel.tables[1] << " was never loaded." << std::endl;
            return;
        }

        const std::set<int> &column_idxs = exec_info.loaded_columns[rel.tables[1]];
        tables[current_table] = loadTable(rel.tables[1], table_column_numbers[rel.tables[1]], column_idxs, queue);
        tables[current_table].table_name = rel.tables[1];
        if (exec_info.group_by_columns.find(rel.tables[1]) != exec_info.group_by_columns.end())
            tables[current_table].group_by_column = exec_info.group_by_columns[rel.tables[1]];
        output_table[rel.id] = current_table;
        current_table++;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int id : exec_info.dag_order)
    {
        const RelNode &rel = result.rels[id];
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            break;
        case RelNodeType::FILTER:
        {
            auto start_filter = std::chrono::high_resolution_clock::now();
            parse_filter(rel.condition, tables[output_table[rel.id - 1]], "", queue);
            output_table[rel.id] = output_table[rel.id - 1];
            auto end_filter = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> filter_time = end_filter - start_filter;
            std::cout << "Filter operation (" << filter_time.count() << " ms)" << std::endl;
            break;
        }
        case RelNodeType::PROJECT:
        {
            auto start_project = std::chrono::high_resolution_clock::now();
            parse_project(rel.exprs, tables[output_table[rel.id - 1]], queue, resources);
            output_table[rel.id] = output_table[rel.id - 1];
            auto end_project = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> project_time = end_project - start_project;
            std::cout << "Project operation (" << project_time.count() << " ms)" << std::endl;
            break;
        }
        case RelNodeType::AGGREGATE:
        {
            auto start_aggregate = std::chrono::high_resolution_clock::now();
            parse_aggregate(tables[output_table[rel.id - 1]], rel.aggs[0], rel.group, queue, resources);
            output_table[rel.id] = output_table[rel.id - 1];
            auto end_aggregate = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> aggregate_time = end_aggregate - start_aggregate;
            std::cout << "Aggregate operation (" << aggregate_time.count() << " ms)" << std::endl;
            break;
        }
        case RelNodeType::JOIN:
        {
            auto start_join = std::chrono::high_resolution_clock::now();
            parse_join(rel, tables[output_table[rel.inputs[0]]], tables[output_table[rel.inputs[1]]], exec_info.table_last_used, queue);
            output_table[rel.id] = output_table[rel.inputs[0]];
            auto end_join = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> join_time = end_join - start_join;
            std::cout << "Join operation (" << join_time.count() << " ms)" << std::endl;
            break;
        }
        case RelNodeType::SORT:
        {
            auto start_sort = std::chrono::high_resolution_clock::now();
            parse_sort(rel, tables[output_table[rel.id - 1]], queue);
            output_table[rel.id] = output_table[rel.id - 1];
            auto end_sort = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> sort_time = end_sort - start_sort;
            std::cout << "Sort operation (" << sort_time.count() << " ms)" << std::endl;
            break;
        }
        default:
            std::cout << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exec_time = end - start;

    // print_result(tables[output_table[result.rels.size() - 1]]);
    save_result(tables[output_table[result.rels.size() - 1]], data_path);

    std::cout << "Execution time: " << exec_time.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

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

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> free_time = end - start;
    std::cout << "Free resources time: " << free_time.count() << " ms" << std::endl;
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
        PlanResult result;
        client.parse(result, sql);

        // std::cout << "Result: " << result << std::endl;

        execute_result(result, argv[1]);
        // execute_result(result, argv[1]);

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