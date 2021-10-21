/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "lm/model.hh"
#include "lm/config.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "util/string_stream.hh"
#include "KenLm.h"
#include <memory>
#include <vector>
#include <string>
#include <stack>
#include <map>
#include <unordered_map>

namespace pnk
{
    static std::vector<std::string> synonym = {"linh", "lẻ", "lẽ", "không", "ngàn", "triệu", "nghìn", "trăm", "tỉ", "tỷ", "một", "hai", "ba", "bốn", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười", "mươi", "tư", "mốt"};
    static std::vector<std::string> others = {"linh", "lẻ", "lẽ"};
    static std::vector<std::string> startWord = {"không", "một", "hai", "ba", "bốn", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười", "triệu", "trăm", "tỉ", "tỷ", "nghìn"};
    static std::map<std::string, int> highUnit = {
        {"tỉ",      1000000000},
        {"tỷ",      1000000000},
        {"triệu",   1000000},
        {"ngàn",    1000},
        {"nghìn",   1000}
    };
    static std::map<std::string, int> lowUnit = {
        {"trăm",    100},
        {"mươi",    10}
    };
    static std::map<std::string, std::string> numWords = {
        {"không",   "0"},
        {"một",     "1"},
        {"mốt",     "1"},
        {"hai",     "2"},
        {"ba",      "3"},
        {"bốn",     "4"},
        {"tư",      "4"},
        {"năm",     "5"},
        {"lăm",     "5"},
        {"sáu",     "6"},
        {"bảy",     "7"},
        {"tám",     "8"},
        {"chín",    "9"},
        {"mười",    "1"}
    };

    class HypsState {
    public:
        HypsState(std::string word, HypsState* parents, bool checked = false, bool root = false)
            : word(word), parents(parents), checked(checked), root(root) {}
        HypsState* parents = nullptr;
        std::vector<std::string> qNum;
        bool nextNum = false;
        std::string word;
        bool root = false;
        uint64_t num = 0;
        bool checked = false;
    };

    class WordToNum {
    public:
        WordToNum(){};
        void run(std::string& sentence, std::shared_ptr<KenLM> lm);
    private:
        uint _maxSym = 1;
        void tokenize(std::string const &str, const char delim,
            std::vector<std::string> &out);
        bool isStartWord(const std::string w);
        bool isSynonym(const std::string w);
        bool isPathOfNum(const std::string w);
        bool isHighUnit(const std::string w);
        bool isLowUnit(const std::string w);
        bool isOthersWord(const std::string w);
        bool checkFormat(std::vector<std::string> queueNum);
        std::string returnRaw(std::vector<std::string> queueNum);
        std::string lessThousandToText(std::vector<std::string> queueNum);
        std::string greaterThousandToText(std::vector<std::string> queueNum);
        std::unordered_map<int, std::vector<HypsState>> hyps_;
        void getBestHypothesis(std::vector<HypsState>& hyps, std::shared_ptr<KenLM> lm);
    };
}