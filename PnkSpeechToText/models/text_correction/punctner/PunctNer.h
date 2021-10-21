
/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "tf.h"
#include <iostream>
#include <dirent.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <sstream>
#include <codecvt>
#include <boost/locale.hpp>
#include <unordered_map>
#include "utils/common.h"
namespace pnk
{
    class PunctuationAndNer
    {
    public:
        PunctuationAndNer(std::string modelPath, bool punct_enable, bool ner_enable);
        ~PunctuationAndNer();
        void predict(std::string &sentence);

    private:
        model_t model_;
        bool _punct_enable = true;
        bool _ner_enable = true;
        std::unordered_map<std::string, int> wordDict_;
        std::unordered_map<std::string, int> charDict_;
        std::unordered_map<int, std::string> puncDict_;
        std::string labelToPunc(const std::string label);
        void tokenize(std::string const &str, const char delim,
            std::vector<std::string> &out);
        bool isNumber(const std::string text);
        std::vector<int32_t> word2vec(std::string word);
        Json::Reader reader;
    };
} // namespace pnk