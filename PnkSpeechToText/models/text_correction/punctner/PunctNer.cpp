/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "PunctNer.h"

namespace pnk
{
    PunctuationAndNer::PunctuationAndNer(std::string modelPath, bool punct_enable, bool ner_enable)
    {
        std::string graphPath, cpPrefix, vocabPath;
        DIR *dir;
        struct dirent *diread;
        _punct_enable = punct_enable;
        _ner_enable = ner_enable;
        if ((dir = opendir(modelPath.c_str())) != nullptr)
        {
            while ((diread = readdir(dir)) != nullptr)
            {
                std::string tmpStr(diread->d_name);
                if (tmpStr.find(".pb") != std::string::npos)
                {
                    graphPath = modelPath + "/" + tmpStr;
                }
                if (tmpStr.find(".index") != std::string::npos)
                {
                    tmpStr.erase(tmpStr.end() - 6, tmpStr.end());
                    cpPrefix = modelPath + "/" + tmpStr;
                }
                if (tmpStr.find(".json") != std::string::npos)
                {
                    vocabPath = modelPath + "/" + tmpStr;
                }
            }
            closedir(dir);
        }
        else
        {
            LOG(WARNING) << "Prediction punctuation model is not existed";
            exit(0);
        }
        if (graphPath.empty() || cpPrefix.empty() || vocabPath.empty())
        {
            LOG(WARNING) << "Cannot find checkpoint, graph file or vocab file";
            exit(0);
        }

        if (!ModelCreate(&model_, graphPath.c_str()))
        {
            LOG(WARNING) << "Creating punctuation prediction is failed";
            exit(0);
        }
        if (!ModelCheckpoint(&model_, cpPrefix.c_str()))
        {
            LOG(WARNING) << "Restoring punctuation prediction checkpoint is falied";
            exit(0);
        }
        // Init word embedding
        std::ifstream vocabFs(vocabPath.c_str());
        
        Json::Value jsonVal;
        reader.parse(vocabFs, jsonVal);
        for (auto const &key : jsonVal["word_dict"].getMemberNames())
        {
            wordDict_.insert(std::pair<std::string, int>(key, jsonVal["word_dict"][key].asUInt()));
        }
        for (auto const &key : jsonVal["char_dict"].getMemberNames())
        {
            charDict_.insert(std::pair<std::string, int>(key, jsonVal["char_dict"][key].asUInt()));
        }
        for (auto const &key : jsonVal["punc_dict"].getMemberNames())
        {
            std::string punc = labelToPunc(key);
            if (!punc.empty())
                puncDict_.insert(std::pair<int, std::string>(jsonVal["punc_dict"][key].asUInt(), punc));
        }
    }
    PunctuationAndNer::~PunctuationAndNer()
    {
        ModelDestroy(&model_);
    }
    std::string PunctuationAndNer::labelToPunc(const std::string label)
    {
        if (label == "O")
            return " ";
        else if (label == "P")
            return ". ";
        else if (label == "C")
            return ", ";
        else if (label == "Q")
            return "? ";
        else
            return "";
    }
    void PunctuationAndNer::tokenize(std::string const &str, const char delim,
                                   std::vector<std::string> &out)
    {
        std::istringstream ss(str);
        std::string token;
        while (std::getline(ss, token, delim))
        {
            out.push_back(token);
        }
    }
    bool PunctuationAndNer::isNumber(const std::string text)
    {
        for (char const &c : text)
        {
            if (std::isdigit(c) == 0)
                if ((c != ',') && (c != '.'))
                    return false;
        }
        return true;
    }
    std::vector<int32_t> PunctuationAndNer::word2vec(std::string word)
    {
        std::vector<int32_t> chars;
        const char *p = word.c_str();
        char *pp = const_cast<char *>(p);
        int ii = 1;
        for (int i = 0; i < word.size(); i++)
        {
            if (charDict_[std::string(pp, ii)])
            {
                chars.push_back(charDict_[std::string(pp, ii)]);
                pp += ii;
                ii = 1;
            }
            else
            {
                ii++;
            }
        }
        return chars;
    }
    void PunctuationAndNer::predict(std::string &sentence)
    {
        std::vector<std::string> words;
        tokenize(sentence, ' ', words);
        int seqLen = words.size();
        uint puncNum = puncDict_.size();
        uint nerNum = 2;// size of ner dictionary;
        uint maxLen = 0;
        std::vector<int32_t> wordEmbedding;
        std::vector<int32_t> charEmbedding;
        std::vector<std::vector<int32_t>> tmpCharEmbedding;
        for (auto word : words)
        {
            uint wordLen = 0;
            // std::vector<int32_t> chars;
            if (wordDict_.find(word) != wordDict_.end())
                wordEmbedding.push_back(wordDict_[word]);
            else if (isNumber(word))
                wordEmbedding.push_back(2);
            else
                wordEmbedding.push_back(1);

            std::vector<int32_t> chars = word2vec(word);
            wordLen = chars.size();
            maxLen = (maxLen > wordLen) ? maxLen : wordLen;
            tmpCharEmbedding.push_back(chars);
        }
        maxLen = std::max(maxLen, (uint)5);
        for (auto &chars : tmpCharEmbedding)
            while (chars.size() < maxLen)
                chars.push_back(0);

        for (int i = 0; i < tmpCharEmbedding.size(); i++)
        {
            for (auto chars : tmpCharEmbedding.at(i))
            {
                if (!isNumber(words.at(i)))
                    charEmbedding.push_back(chars);
                else
                    charEmbedding.push_back(1);
            }
        }
        uint len = wordEmbedding.size();
        feed_dict_t feedDict;

        feedDict.words = wordEmbedding.data();
        feedDict.chars = charEmbedding.data();
        feedDict.drop_rate = 0.0;
        feedDict.is_train = 0;
        feedDict.seq_len = len;
        feedDict.max_word_len = maxLen;
        float *predicts = new float[len * (nerNum + puncNum)];
        if (!ModelPredict(&model_, feedDict, predicts))
        {
            LOG(WARNING) << "Prediction punctuation is failed";
            // Return default predict
            if (!predicts)
                predicts = (float*)malloc(len * (nerNum + puncNum) * sizeof(float));
            for (int i = 0; i < len * (nerNum + puncNum); i++)
                predicts[i] = 1;
        }
        // Capitalize first letter of first word
        boost::locale::generator gen;
        std::locale loc = gen("vi-VN.UTF-8");
        std::locale::global(loc);
        std::string out = "";
        bool newSen = true;
        for (int i = 0; i < len; i++)
        {
            int puncMaxId = 0;
            int nerMaxId = 0;
            float maxNer = 0;
            float maxPunc = predicts[(i * (puncNum + nerNum)) + puncMaxId];
            
            for (int j = 1; j < puncNum; j++)
            {
                if (j == puncNum)
                    continue;
                float puncAcc = predicts[(i * (puncNum + nerNum)) + j];
                if (puncAcc > maxPunc)
                {
                    puncMaxId = j;
                    maxPunc = puncAcc;
                }
            }
            if (predicts[(i * (puncNum + nerNum)) + puncNum] > predicts[(i * (puncNum + nerNum)) + puncNum + 1])
            {
                maxNer = predicts[(i * (puncNum + nerNum)) + puncNum];
                nerMaxId = 0;
            } else {
                maxNer = predicts[(i * (puncNum + nerNum)) + puncNum + 1];
                nerMaxId = 1;
            }
            if ((maxPunc < 0.7) || (!_punct_enable))
                puncMaxId = 0;
            if ((maxNer < 0.7) || (!_ner_enable))
                nerMaxId = 0;
            if (newSen)
            {
                out += boost::locale::to_title(words.at(i)) + ((i != (len - 1)) ? puncDict_[puncMaxId] : ".");
                newSen = false;
            }
            else
            {
                if (nerMaxId == 1)
                    out += boost::locale::to_title(words.at(i)) + ((i != (len - 1)) ? puncDict_[puncMaxId] : ".");
                else
                    out += words.at(i) + ((i != (len - 1)) ? puncDict_[puncMaxId] : ".");
            }
            if ((puncDict_[puncMaxId] == ". ") || ((puncDict_[puncMaxId] == "? ")))
                newSen = true;
        }
        sentence = out;
        delete[] predicts;
    }
} // namespace pnk
