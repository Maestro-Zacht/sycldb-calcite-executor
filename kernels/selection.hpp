#pragma once

#include <sycl/sycl.hpp>

enum comp_op
{
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE
};
enum logical_op
{
    NONE,
    AND,
    OR
};

comp_op get_comp_op(std::string op)
{
    if (op == "==")
        return EQ;
    else if (op == ">=")
        return GE;
    else if (op == "<=")
        return LE;
    else if (op == "<")
        return LT;
    else if (op == ">")
        return GT;
    else if (op == "!=" || op == "<>")
        return NE;
    else
        return EQ;
}

logical_op get_logical_op(std::string op)
{
    if (op == "AND")
        return AND;
    else if (op == "OR")
        return OR;
    else
        return NONE;
}

template <typename T>
inline bool compare(comp_op CO, T a, T b)
{
    switch (CO)
    {
    case EQ:
        return a == b;
    case NE:
        return a != b;
    case LT:
        return a < b;
    case LE:
        return a <= b;
    case GT:
        return a > b;
    case GE:
        return a >= b;
    default:
        return false;
    }
}

inline bool logical(logical_op logic, bool a, bool b)
{
    switch (logic)
    {
    case AND:
        return a && b;
    case OR:
        return a || b;
    case NONE:
        return b;
    default:
        return false;
    }
}

template <typename T>
void selection(bool flags[], T arr[], std::string op, T value, std::string parent_op, int col_len, sycl::queue &queue)
{
    comp_op comparison = get_comp_op(op);
    logical_op logic = get_logical_op(parent_op);

    // for (int i = 0; i < col_len; i++)
    //     flags[i] = logical(logic, flags[i], compare(comparison, arr[i], value));

    queue.parallel_for(
        col_len,
        [=](sycl::id<1> idx)
        {
            flags[idx] = logical(logic, flags[idx], compare(comparison, arr[idx], value));
        }
    );
    queue.wait();

    // std::cout << "Running selection with comparison: " << op << " and parent op " << parent_op << std::endl;
}

template <typename T>
void selection(bool flags[], T operand1[], std::string op, T operand2[], std::string parent_op, int col_len, sycl::queue &queue)
{
    comp_op comparison = get_comp_op(op);
    logical_op logic = get_logical_op(parent_op);

    // for (int i = 0; i < col_len; i++)
    //     flags[i] = logical(logic, flags[i], compare(comparison, operand1[i], operand2[i]));

    queue.parallel_for(
        col_len,
        [=](sycl::id<1> idx)
        {
            flags[idx] = logical(logic, flags[idx], compare(comparison, operand1[idx], operand2[idx]));
        }
    );
    queue.wait();

    // std::cout << "Running selection with comparison: " << op << " and parent op " << parent_op << std::endl;
}