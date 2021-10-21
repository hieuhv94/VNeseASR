/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include <regex>
#include <string>
#include <iostream>
#include <fstream>

namespace pnk {
static std::vector<std::pair<std::string, std::string>> regRules = 
{
    {"\\b([0-9]+) phần trăm", "$1%"},
    {"âm ([0-9]+)", "-$1"},
    {"\\b([0-9]+) phẩy ([0-9]+)", "$1,$2"},
    {"ngày( | mùng | mồng )([1-9]|[12][0-9]|3[01]) tháng ([1-9]|1[0-2]) năm ([0-9]+)", "ngày $2/$3/$4"},
    {"ngày( | mùng | mồng )([1-9]|[12][0-9]|3[01]) tháng ([1-9]|1[0-2])", "ngày $2/$3"},
    {"tháng ([1-9]|1[0-2]) năm ([0-9]+)", "tháng $1/$2"},
    {" 0 ([0-9]+)", " 0$1"},
    {"(^0 | 0$| 0 )", " không "},
    {"([0-9]+)000000000", "$1 tỷ"},
    {"([0-9]+)000000", "$1 triệu"},
    {",([0-9]+)000", ",$1 nghìn"},
    {"([0-9]+) độ xê", "$1 độ c"},
    {"([0-9]+) trên ([0-9]+)", "$1/$2"},
    {"([0-9]+) phần ([0-9]+)", "$1/$2"}
};

void applyRules(std::string& text)
{
    for (const std::pair<std::string, std::string> rule : regRules)
    {
        text = std::regex_replace(text, std::regex(rule.first), rule.second);
    }
}
}