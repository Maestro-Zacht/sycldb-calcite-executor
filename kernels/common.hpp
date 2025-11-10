#pragma once

class KernelDefinition
{
private:
    int col_len;
public:
    KernelDefinition(int col_len) : col_len(col_len) {}
    int get_col_len() const { return col_len; }
};