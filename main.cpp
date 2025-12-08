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
#include "operations/memory_manager.hpp"

#include "models/models.hpp"
#include "models/transient_table.hpp"

#include "kernels/types.hpp"

#include "common.hpp"

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

class InitTimer1;
class InitTimer2;
class InitTimer3;
class EndTimer1;
class EndTimer2;
class EndTimer3;

void sycl_exception_handler(sycl::exception_list exceptions)
{
    bool error = false;
    for (const auto &e : exceptions)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (sycl::exception &e)
        {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            error = true;
        }
    }

    if (error)
        std::terminate();
}

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

void save_result(const TransientTable &table, const std::string &data_path)
{
    std::string query_name = data_path.substr(data_path.find_last_of("/") + 1, 3);
    std::cout << "Saving result to " << query_name << ".res" << std::endl;

    std::ofstream outfile(query_name + ".res");
    if (!outfile.is_open())
    {
        std::cerr << "Could not open result file for writing." << std::endl;
        return;
    }

    outfile << table;

    outfile.close();
}

std::chrono::duration<double, std::milli> execute_result(
    const PlanResult &result,
    const std::string &data_path,
    const std::map<std::string,
    TableData<int>> &all_tables,
    sycl::queue &queue,
    memory_manager &gpu_allocator,
    std::ostream &perf_out = std::cout
)
{
    #if PERFORMANCE_MEASUREMENT_ACTIVE
    bool output_done = false;
    #endif

    #if USE_FUSION
    sycl::ext::codeplay::experimental::fusion_wrapper fw{ queue };
    bool fusion_active = false;
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
        tables[current_table] = copy_table(all_tables.at(rel.tables[1]), column_idxs, gpu_allocator, queue);

        if (exec_info.group_by_columns.find(rel.tables[1]) != exec_info.group_by_columns.end())
            tables[current_table].group_by_column = exec_info.group_by_columns[rel.tables[1]];
        output_table[rel.id] = current_table;

        if (rel.tables[1] != "lineorder" && exec_info.prepare_join.find(rel.tables[1]) != exec_info.prepare_join.end())
        {
            auto [join_id, right_column_idx] = exec_info.prepare_join[rel.tables[1]];

            TableData<int> &table_info = tables[current_table];
            ColumnData<int> &column_info = table_info.columns[table_info.column_indices.at(right_column_idx)];

            queue.wait();

            if (exec_info.table_last_used[rel.tables[1]] == join_id)
            {
                // filter join ht
                int ht_len = column_info.max_value - column_info.min_value + 1;
                bool *ht = gpu_allocator.alloc<bool>(ht_len, true);

                table_info.ht = ht;
                table_info.ht_min = column_info.min_value;
                table_info.ht_max = column_info.max_value;
            }
            else
            {
                // full join ht
                int ht_len = column_info.max_value - column_info.min_value + 1,
                    *ht = gpu_allocator.alloc<int>(ht_len * 2, true);

                table_info.ht = ht;
                table_info.ht_min = column_info.min_value;
                table_info.ht_max = column_info.max_value;
            }
        }

        current_table++;
    }

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Execution order: ";
    for (int id : exec_info.dag_order)
        std::cout << id << " -> ";
    std::cout << std::endl;
    #endif

    queue.wait();

    queue.single_task<InitTimer1>([=]() {}).wait();

    auto start = std::chrono::high_resolution_clock::now();

    for (int id : exec_info.dag_order)
    {
        const RelNode &rel = result.rels[id];
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            dependencies[rel.id] = {};

            #if USE_FUSION
            if (rel.tables[1] == "lineorder")
            {
                fw.start_fusion();
                fusion_active = true;
            }
            #endif

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
                gpu_allocator,
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
                gpu_allocator,
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
            #if USE_FUSION
            if (fusion_active)
            {
                fw.complete_fusion(sycl::ext::codeplay::experimental::property::no_barriers {});
                fusion_active = false;
            }
            #endif

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto start_aggregate = std::chrono::high_resolution_clock::now();
            #endif

            dependencies[rel.id] = parse_aggregate(
                tables[output_table[rel.id - 1]],
                rel.aggs[0],
                rel.group,
                resources,
                gpu_allocator,
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
                gpu_allocator,
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
            #if USE_FUSION
            if (fusion_active)
            {
                fw.complete_fusion(sycl::ext::codeplay::experimental::property::no_barriers {});
                fusion_active = false;
            }
            #endif

            queue.wait();
            queue.single_task<EndTimer1>([=]() {}).wait();

            auto start_sort = std::chrono::high_resolution_clock::now();
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            parse_sort(rel, tables[output_table[rel.id - 1]], queue);
            #endif
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

        if (exec_info.prepare_join_id.find(rel.id) != exec_info.prepare_join_id.end())
        {
            auto [join_id, right_column_idx] = exec_info.prepare_join_id[rel.id];

            TableData<int> &table_info = tables[output_table[rel.id]];
            ColumnData<int> &column_info = table_info.columns[table_info.column_indices.at(right_column_idx)];

            std::vector<sycl::event> &ht_dependencies = dependencies[rel.id];

            if (table_info.ht == nullptr)
            {
                std::cerr << "!!!!! Table " << table_info.table_name
                    << " does not have a hash table !!!!!" << std::endl;
                continue;
            }

            if (join_id == exec_info.table_last_used[table_info.table_name])
            {
                // filter join ht
                int ht_len = table_info.ht_max - table_info.ht_min + 1;
                bool *ht = (bool *)table_info.ht;

                auto e2 = build_keys_ht(column_info.content, table_info.flags, table_info.col_len, ht, ht_len, column_info.min_value, queue, ht_dependencies);
                ht_dependencies = { e2 };
            }
            else
            {
                // full join ht
                const ColumnData<int> &group_by_column_info = table_info.columns[table_info.column_indices.at(table_info.group_by_column)];
                int ht_len = table_info.ht_max - table_info.ht_min + 1,
                    *ht = (int *)table_info.ht;

                auto e2 = build_key_vals_ht(column_info.content, group_by_column_info.content, table_info.flags, table_info.col_len, ht, ht_len, column_info.min_value, queue, ht_dependencies);
                ht_dependencies = { e2 };
            }
        }

        if (rel.relOp != RelNodeType::TABLE_SCAN)
        {
            std::cout << "rows selected after operation " << id << ": "
                << count_true_flags(tables[output_table[id]].flags, tables[output_table[id]].col_len, queue, dependencies[id])
                << std::endl;
        }
    }

    #if USE_FUSION
    if (fusion_active)
        fw.complete_fusion(sycl::ext::codeplay::experimental::property::no_barriers {});
    #endif

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

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    TableData<int> &final_table = tables[output_table[result.rels.size() - 1]];
    memory_manager final_table_allocator(queue, ((uint64_t)1) << 30);
    for (int i = 0; i < final_table.columns_size; i++)
    {
        if (final_table.columns[i].has_ownership)
        {
            if (final_table.columns[i].is_aggregate_result)
            {
                uint64_t *host_col = final_table_allocator.alloc<uint64_t>(final_table.col_len, false);
                queue.copy((uint64_t *)final_table.columns[i].content, host_col, final_table.col_len).wait();
                final_table.columns[i].content = (int *)host_col;
            }
            else
            {
                int *host_col = final_table_allocator.alloc<int>(final_table.col_len, false);
                queue.copy(final_table.columns[i].content, host_col, final_table.col_len).wait();
                final_table.columns[i].content = host_col;
            }
        }
        else
            std::cout << "!!!!!!!!!! Column " << i << " does not have ownership, skipping copy to host !!!!!!!!!!" << std::endl;
    }

    bool *host_flags = final_table_allocator.alloc<bool>(final_table.col_len, false);
    queue.copy(final_table.flags, host_flags, final_table.col_len).wait();
    final_table.flags = host_flags;

    // print_result(final_table);
    save_result(final_table, data_path);

    start = std::chrono::high_resolution_clock::now();
    #endif

    for (int i = 0; i < current_table; i++)
    {
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

int normal_execution(int argc, char **argv)
{
    std::shared_ptr<TTransport> socket(new TSocket("localhost", 5555));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    CalciteServerClient client(protocol);
    std::string sql;
    sycl::queue queue{
        sycl::gpu_selector_v,
        #if USE_FUSION
        sycl::ext::codeplay::experimental::property::queue::enable_fusion {}
        #endif
    };
    memory_manager table_allocator(queue, SIZE_TEMP_MEMORY_CPU); // memory manager for table allocations (on host)
    memory_manager gpu_allocator(queue, SIZE_TEMP_MEMORY_GPU); // memory manager for temporary allocations during query execution

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    #endif

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

    auto all_tables = preload_all_tables(queue, table_allocator);

    try
    {
        // std::cout << "SQL Query: " << sql << std::endl;
        transport->open();
        std::cout << "Transport opened successfully." << std::endl;

        #if PERFORMANCE_MEASUREMENT_ACTIVE
        std::string sql_filename = argv[1];
        std::string query_name = sql_filename.substr(sql_filename.find_last_of("/") + 1, 3);
        std::ofstream perf_file(query_name + "-performance-cpu-s100.log", std::ios::out | std::ios::trunc);
        if (!perf_file.is_open())
        {
            std::cerr << "Could not open performance log file: " << query_name << "-performance-cpu-s100.log" << std::endl;
            return 1;
        }

        for (int i = 0; i < PERFORMANCE_REPETITIONS; i++)
        {
            PlanResult result;

            client.parse(result, sql);
            // std::cout << "Starting repetition " << i + 1 << "/" << PERFORMANCE_REPETITIONS << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto exec_time = execute_result(result, argv[1], all_tables, queue, gpu_allocator);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> total_time = end - start;

            std::cout << "Repetition " << i + 1 << "/" << PERFORMANCE_REPETITIONS
                << " - " << exec_time.count() << " ms - "
                << total_time.count() << " ms" << std::endl;
            perf_file << total_time.count() << '\n';
            gpu_allocator.reset();
        }
        perf_file.close();
        #else
        PlanResult result;
        client.parse(result, sql);

        // std::cout << "Result: " << result << std::endl;

        execute_result(result, argv[1], all_tables, queue, gpu_allocator);
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

std::chrono::duration<double, std::milli> ddor_execute_result(
    const PlanResult &result,
    const std::string &data_path,
    Table tables[MAX_NTABLES],
    sycl::queue &gpu_queue,
    sycl::queue &cpu_queue,
    #if USE_FUSION
    sycl::ext::codeplay::experimental::fusion_wrapper &fw_gpu,
    sycl::ext::codeplay::experimental::fusion_wrapper &fw_cpu,
    #endif
    memory_manager &gpu_allocator,
    memory_manager &cpu_allocator,
    std::ostream &perf_out = std::cout)
{

    ExecutionInfo exec_info = parse_execution_info(result);
    std::vector<int> output_table(result.rels.size(), -1);
    std::vector<TransientTable> transient_tables;

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

        Table *table_ptr = nullptr;
        for (int i = 0; i < MAX_NTABLES; i++)
        {
            if (tables[i].get_name() == rel.tables[1])
            {
                table_ptr = &tables[i];
                break;
            }
        }

        if (table_ptr == nullptr)
        {
            std::cerr << "Table " << rel.tables[1] << " not found among loaded tables." << std::endl;
            return std::chrono::duration<double, std::milli>::zero();
        }

        TransientTable &t = transient_tables.emplace_back(
            table_ptr,
            gpu_queue,
            cpu_queue,
            #if USE_FUSION
            fw_gpu,
            fw_cpu,
            #endif
            gpu_allocator,
            cpu_allocator
        );

        if (exec_info.group_by_columns.find(rel.tables[1]) != exec_info.group_by_columns.end())
            t.set_group_by_column(exec_info.group_by_columns[rel.tables[1]]);

        output_table[rel.id] = transient_tables.size() - 1;
    }

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Execution order: ";
    for (int id : exec_info.dag_order)
        std::cout << id << " -> ";
    std::cout << std::endl;
    #endif

    gpu_queue.wait();
    cpu_queue.wait();

    gpu_queue.single_task<InitTimer2>([=]() {}).wait();
    cpu_queue.single_task<InitTimer3>([=]() {}).wait();

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
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Starting Filter operation." << std::endl;
            auto start_filter = std::chrono::high_resolution_clock::now();
            #endif
            int prev_table_idx = output_table[id - 1];
            transient_tables[prev_table_idx].apply_filter(
                rel.condition,
                "",
                gpu_allocator,
                cpu_allocator
            );
            output_table[id] = prev_table_idx;
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
            std::cout << "Starting Project operation." << std::endl;
            auto start_project = std::chrono::high_resolution_clock::now();
            #endif
            int prev_table_idx = output_table[id - 1];
            transient_tables[prev_table_idx].apply_project(
                rel.exprs,
                gpu_allocator,
                cpu_allocator
            );
            output_table[id] = prev_table_idx;
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
            std::cout << "Starting Aggregate operation." << std::endl;
            auto start_aggregate = std::chrono::high_resolution_clock::now();
            #endif
            int prev_table_idx = output_table[id - 1];
            transient_tables[prev_table_idx].apply_aggregate(
                rel.aggs[0],
                rel.group,
                gpu_allocator,
                cpu_allocator
            );
            output_table[id] = prev_table_idx;
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
            std::cout << "Starting Join operation." << std::endl;
            auto start_join = std::chrono::high_resolution_clock::now();
            #endif

            int left_table_idx = output_table[rel.inputs[0]];
            int right_table_idx = output_table[rel.inputs[1]];

            transient_tables[left_table_idx].apply_join(
                transient_tables[right_table_idx],
                rel,
                gpu_allocator,
                cpu_allocator
            );
            output_table[id] = left_table_idx;

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            auto end_join = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> join_time = end_join - start_join;
            std::cout << "Join operation (" << join_time.count() << " ms)" << std::endl;
            #endif

            break;
        }
        case RelNodeType::SORT:
        {
            // Sort is done in bash
            output_table[id] = output_table[id - 1];
            break;
        }
        default:
            std::cerr << "RelNodeType " << rel.relOp << " not yet supported in DDOR." << std::endl;
            break;
        }

        // if (rel.relOp != RelNodeType::TABLE_SCAN)
        // {
        //     uint64_t rows_selected_gpu = transient_tables[output_table[id]].count_flags_true(true, gpu_allocator, cpu_allocator),
        //         rows_selected_cpu = transient_tables[output_table[id]].count_flags_true(false, gpu_allocator, cpu_allocator);
        //     std::cout << "== rows selected after operation " << id
        //         << "\n==== GPU: "
        //         << rows_selected_gpu
        //         << "\n==== CPU: "
        //         << rows_selected_cpu
        //         << std::endl;
        // }
    }

    transient_tables[output_table[result.rels.size() - 1]].execute_pending_kernels();

    // auto pre_wait = std::chrono::high_resolution_clock::now();

    gpu_queue.wait();
    cpu_queue.wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    // std::chrono::duration<double, std::milli> wait_time = end - pre_wait;

    // std::cout << "All operations completed in " << duration.count() << " ms" << std::endl;

    gpu_queue.single_task<EndTimer2>([=]() {}).wait();
    cpu_queue.single_task<EndTimer3>([=]() {}).wait();

    // std::cout << "end" << std::endl;

    #if PERFORMANCE_MEASUREMENT_ACTIVE
    perf_out << duration.count() << '\n';
    #else
    TransientTable &final_table = transient_tables[output_table[result.rels.size() - 1]];

    final_table.update_flags(false, gpu_allocator, cpu_allocator);
    auto events = final_table.execute_pending_kernels();
    sycl::event::wait(events.first);
    sycl::event::wait(events.second);
    gpu_queue.wait_and_throw();
    cpu_queue.wait_and_throw();
    final_table.assert_flags_to_cpu();
    // std::cout << "Final result:\n" << final_table << std::endl;
    save_result(final_table, data_path);
    gpu_queue.wait_and_throw();
    cpu_queue.wait_and_throw();
    #endif

    return duration;
}

int data_driven_operator_replacement(int argc, char **argv)
{
    std::shared_ptr<TTransport> socket(new TSocket("localhost", 5555));
    std::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    CalciteServerClient client(protocol);
    std::string sql;
    sycl::queue gpu_queue{
        sycl::gpu_selector_v,
        // sycl_exception_handler,
        #if USE_FUSION
        sycl::ext::codeplay::experimental::property::queue::enable_fusion {}
        #endif
    }, cpu_queue{
        sycl::cpu_selector_v,
        // sycl_exception_handler,
        #if USE_FUSION
        sycl::ext::codeplay::experimental::property::queue::enable_fusion {}
        #endif
    };

    #if USE_FUSION
    sycl::ext::codeplay::experimental::fusion_wrapper fw_gpu{ gpu_queue }, fw_cpu{ cpu_queue };
    #endif

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

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Running on GPU: " << gpu_queue.get_device().get_info<sycl::info::device::name>()
        << "\nRunning on CPU: " << cpu_queue.get_device().get_info<sycl::info::device::name>()
        << std::endl;
    #endif

    Table tables[MAX_NTABLES] = {
        Table("part", gpu_queue, cpu_queue),
        Table("supplier", gpu_queue, cpu_queue),
        Table("customer", gpu_queue, cpu_queue),
        Table("ddate", gpu_queue, cpu_queue),
        Table("lineorder", gpu_queue, cpu_queue),
    };

    for (const Table &table : tables)
    {
        std::cout << table.get_name() << " num segments: " << table.num_segments() << std::endl;
    }

    // tables[0].move_column_to_device(0);
    // tables[0].move_column_to_device(4);

    // tables[1].move_column_to_device(0);
    // tables[1].move_column_to_device(4);

    // tables[2].move_column_to_device(0);
    // tables[2].move_column_to_device(4);

    // tables[3].move_column_to_device(0);
    // tables[3].move_column_to_device(4);

    // tables[4].move_column_to_device(2);
    // tables[4].move_column_to_device(3);
    // tables[4].move_column_to_device(4);
    // tables[4].move_column_to_device(5);
    // tables[4].move_column_to_device(8);
    // tables[4].move_column_to_device(12);
    // tables[4].move_column_to_device(13);
    // tables[4].move_column_to_device(14);
    // for (int i = 0; i < MAX_NTABLES; i++)
    //     tables[i].move_all_to_device();
    // gpu_queue.wait_and_throw();

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    // std::cout << "All tables moved to device." << std::endl;

    uint64_t total_mem = 0, total_gpu_mem = 0;
    for (int i = 0; i < MAX_NTABLES; i++)
    {
        total_mem += tables[i].get_data_size(false);
        total_gpu_mem += tables[i].get_data_size(true);
    }
    std::cout << "Total memory used by tables: " << (total_mem >> 20)
        << " MB (GPU: " << (total_gpu_mem >> 20) << " MB)" << std::endl;

    #endif

    memory_manager gpu_allocator(gpu_queue, SIZE_TEMP_MEMORY_GPU);
    memory_manager cpu_allocator(cpu_queue, SIZE_TEMP_MEMORY_CPU);

    try
    {
        // std::cout << "SQL Query: " << sql << std::endl;
        transport->open();
        std::cout << "Transport opened successfully." << std::endl;

        #if PERFORMANCE_MEASUREMENT_ACTIVE
        std::string sql_filename = argv[1];
        std::string query_name = sql_filename.substr(sql_filename.find_last_of("/") + 1, 3);
        std::ofstream perf_file(query_name + "-performance-cpu-s100.log", std::ios::out | std::ios::trunc);
        if (!perf_file.is_open())
        {
            std::cerr << "Could not open performance log file: " << query_name << "-performance-cpu-s100.log" << std::endl;
            return 1;
        }

        for (int i = 0; i < PERFORMANCE_REPETITIONS; i++)
        {
            PlanResult result;

            auto start = std::chrono::high_resolution_clock::now();
            client.parse(result, sql);
            auto exec_time = ddor_execute_result(
                result,
                argv[1],
                tables,
                gpu_queue,
                cpu_queue,
                #if USE_FUSION
                fw_gpu,
                fw_cpu,
                #endif
                gpu_allocator,
                cpu_allocator,
                perf_file
            );
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> total_time = end - start;

            gpu_allocator.reset();
            cpu_allocator.reset();

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> after_reset = end - start;

            std::cout << "Repetition " << i + 1 << "/" << PERFORMANCE_REPETITIONS
                << " - " << exec_time.count() << " ms - "
                << total_time.count() << " ms - " << after_reset.count() << " ms" << std::endl;
            // perf_file << total_time.count() << '\n';

            gpu_queue.wait_and_throw();
            cpu_queue.wait_and_throw();
            // sleep(1);
        }
        perf_file.close();
        #else
        PlanResult result;
        client.parse(result, sql);

        auto time = ddor_execute_result(
            result,
            argv[1],
            tables,
            gpu_queue,
            cpu_queue,
            #if USE_FUSION
            fw_gpu,
            fw_cpu,
            #endif
            gpu_allocator,
            cpu_allocator
        );
        std::cout << "DDOR execution completed in " << time.count() << " ms." << std::endl;

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

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Finished execution." << std::endl;
    #endif

    gpu_queue.wait_and_throw();
    cpu_queue.wait_and_throw();

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Finished waiting" << std::endl;
    #endif

    return 0;
}

int test()
{
    sycl::queue gpu_queue{
        sycl::gpu_selector_v,
        #if USE_FUSION
        sycl::ext::codeplay::experimental::property::queue::enable_fusion {}
        #endif
    }, cpu_queue{
        sycl::cpu_selector_v,
        #if USE_FUSION
        sycl::ext::codeplay::experimental::property::queue::enable_fusion {}
        #endif
    };

    uint64_t N = ((uint64_t)1 << 30),
        reps = 50;

    int *data_gpu = sycl::malloc_device<int>(N, gpu_queue);
    int *data_cpu = sycl::malloc_device<int>(N, cpu_queue);

    sycl::event gpu_event = gpu_queue.fill<int>(data_gpu, 1, N),
        cpu_event = cpu_queue.fill<int>(data_cpu, 2, N);

    cpu_event.wait();
    gpu_event.wait();

    for (int i = 0; i < reps; i++)
    {
        auto e1 = gpu_queue.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(gpu_event);
                cgh.parallel_for(
                    sycl::range<1>{N},
                    [=](sycl::id<1> idx)
                    {
                        data_gpu[idx] += 1;
                    }
                );
            }
        );
        gpu_event = e1;

        auto e2 = cpu_queue.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(cpu_event);
                cgh.parallel_for(
                    sycl::range<1>{N},
                    [=](sycl::id<1> idx)
                    {
                        data_cpu[idx] += 1;
                    }
                );
            }
        );
        cpu_event = e2;
    }



    sycl::free(data_gpu, gpu_queue);
    sycl::free(data_cpu, cpu_queue);

    return 0;
}

int main(int argc, char **argv)
{
    // int r = test();
    // int r = normal_execution(argc, argv);
    int r = data_driven_operator_replacement(argc, argv);

    #if not PERFORMANCE_MEASUREMENT_ACTIVE
    std::cout << "Return code: " << r << std::endl;
    #endif
    return r;
}