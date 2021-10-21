/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "WordToNum.h"
#include "Regex.h"
#include <math.h>
namespace pnk
{
    void WordToNum::tokenize(std::string const &str, const char delim,
                             std::vector<std::string> &out)
    {
        size_t start;
        size_t end = 0;

        while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
        {
            end = str.find(delim, start);
            out.push_back(str.substr(start, end - start));
        }
    }
    bool WordToNum::isStartWord(const std::string w)
    {
        if (std::find(startWord.begin(), startWord.end(), w) == startWord.end())
            return false;
        return true;
    }
    bool WordToNum::isSynonym(const std::string w)
    {
        if (std::find(synonym.begin(), synonym.end(), w) == synonym.end())
            return false;
        return true;
    }
    bool WordToNum::isPathOfNum(const std::string w)
    {
        if (numWords.find(w) == numWords.end())
            return false;
        return true;
    }
    bool WordToNum::isHighUnit(const std::string w)
    {
        if (highUnit.find(w) == highUnit.end())
            return false;
        return true;
    }
    bool WordToNum::isLowUnit(const std::string w)
    {
        if (lowUnit.find(w) == lowUnit.end())
            return false;
        return true;
    }
    bool WordToNum::isOthersWord(const std::string w)
    {
        if (std::find(others.begin(), others.end(), w) == others.end())
            return false;
        return true;
    }
    bool WordToNum::checkFormat(std::vector<std::string> queueNum)
    {
        for (auto w : queueNum)
            if (isPathOfNum(w))
                return true;
        return false;
    }
    std::string WordToNum::returnRaw(std::vector<std::string> queueNum)
    {
        std::string res = "";
        for (auto w : queueNum)
            res += w + " ";
        return res;
    }
    inline std::string numberToText(std::string number[], int len)
    {
        std::string sNum = "";
        // for (int i =0; i< 4; i++)
        // std::cout << number[i] << std::endl;
        int rank = -1;
        for (int i = len - 1; i >= 0; i--)
        {
                if (!sNum.empty())
                {
                    if (rank > 1)
                        sNum += ".";
                    if (number[i] == "")
                        sNum += "000";
                    else if (number[i].size() < 2)
                        sNum += "00";
                    else if (number[i].size() < 3)
                        sNum += "0";
                }
                if (!number[i].empty())
                {
                    if (rank == -1)
                        rank = i;
                    sNum += number[i];
                }
        }
        return sNum;
    }
    std::string WordToNum::lessThousandToText(std::vector<std::string> queueNum)
    {
        if (queueNum.empty())
            return "";
        bool wrongFormat = false;
        bool lessHundred = false;
        std::string sNum = "";
        uint minRank = -1;
        for (auto word : queueNum)
        {
            if (isLowUnit(word))
            {
                if (word == "trăm")
                {
                    lessHundred = true;
                    if (sNum.size() == 0)
                    {
                        sNum += '1';
                        minRank = 3;
                    } else if ((sNum.size() > 1) || (minRank != -1))
                    {
                        wrongFormat = true;
                        break;
                    } else {
                        minRank = 3;
                    }
                } else {
                    lessHundred = true;
                    if ((sNum.size() == 0) || (minRank < 3) || (sNum.size() > 3))
                    {
                        wrongFormat = true;
                        break;
                    } else
                    {
                        if (minRank == -1)
                        {
                            minRank = sNum.size() + 1;
                        }
                    }
                }
            }
            else if (isOthersWord(word))
            {
                lessHundred = true;
                if (sNum.size() != 1)
                {
                    wrongFormat =true;
                    break;
                } else {
                    sNum += "0";
                }
            }
            else
            {
                if ((word == "mốt") || (word == "tư"))
                {
                    if (sNum.size() == 0)
                    {
                        wrongFormat = true;
                        break;
                    }
                }
                sNum += numWords[word];
                if (word == "mười")
                {
                    if ((sNum.size() > 1) || (minRank == 2) || (queueNum.size() > 2))
                    {
                        wrongFormat = true;
                        break;
                    }
                    if (minRank == -1)
                        minRank = 2;
                }
            }
        }
        if (wrongFormat)
        {
            std::string ss = queueNum.at(0);
            for (int i = 1; i < queueNum.size(); i++)
                ss += " " + queueNum.at(i);
            return ss;
        }
        if ((sNum.size() < minRank) && (minRank != -1))
        {
            int len = sNum.size();
            for (int i = 0; i < minRank - len; i++)
            {
                sNum += "0";
            }
        }
        return sNum;
    }

    std::string WordToNum::greaterThousandToText(std::vector<std::string> queueNum)
    {
        if (queueNum.empty())
            return "";
        if (!checkFormat(queueNum))
            return returnRaw(queueNum);
        uint64_t value = 0;
        bool isUnit = false;
        std::string number[4] = {"", "", "", ""};
        std::vector<std::string> qNum;
        uint minRank = 4;

        bool isSpec = false;
        for (int i = 0; i < queueNum.size(); ++i)
        {
            std::string word = queueNum.at(i);
            if (isPathOfNum(word) || isLowUnit(word))
            {
                qNum.push_back(word);
            } else if (isOthersWord(word))
            {
                if (!qNum.empty())
                    qNum.push_back(word);
                else
                    isSpec = true;
            }
            else if (isHighUnit(word)) {
                isSpec = false;
                uint rank = ((int)log10(highUnit[word]) / 3);
                if (!qNum.empty())
                {
                    if (rank >= minRank)
                    {
                        std::string num = numberToText(number, 4);
                        if (!num.empty())
                            return num + " " + lessThousandToText(qNum) + " " + word + " " + greaterThousandToText(std::vector<std::string>(queueNum.begin() + i + 1, queueNum.end()));
                        else
                            return lessThousandToText(qNum) + " " + word + " " + greaterThousandToText(std::vector<std::string>(queueNum.begin() + i + 1, queueNum.end()));
                    } else {
                        std::string lessThousand = lessThousandToText(qNum);
                        if (lessThousand.size() < 4)
                        {
                            number[rank] =  lessThousand;
                            minRank = rank;
                        }
                        else
                        {
                            std::string num = numberToText(number, 4);
                            if (!num.empty())
                                return num + " " + lessThousand + " " + word + " " + greaterThousandToText(std::vector<std::string>(queueNum.begin() + i + 1, queueNum.end()));
                            else
                                return lessThousand + " " + word + " " + greaterThousandToText(std::vector<std::string>(queueNum.begin() + i + 1, queueNum.end()));
                        }
                    }
                }
                else {
                    if (i == 0)
                    {
                        minRank = rank;
                        number[rank] = "1"; 
                    } else {
                        std::string num = numberToText(number, 4);
                        if (!num.empty())
                            return num + " " + word + " " + greaterThousandToText(std::vector<std::string>(queueNum.begin() + i + 1, queueNum.end()));
                        else
                            return word + " " + greaterThousandToText(std::vector<std::string>(queueNum.begin() + i + 1, queueNum.end()));
                    }
                }
                qNum.clear();
            }
        }
        if (!qNum.empty())
        {
            std::string lessThousand = lessThousandToText(qNum);
            if (lessThousand.size() < 4)
            {
                if ((lessThousand.size() == 1) && (!isSpec) && (minRank != 4))
                {
                    minRank -= 1;
                    number[minRank] = lessThousand + "00";
                } else {
                    minRank = 0;
                    number[0] = lessThousand;
                }
            } else {
                std::string num = numberToText(number, 4);
                if (!num.empty())
                    return num + " " + lessThousand;
                else
                {
                    return lessThousand;
                }
            }
        }
        std::string res =  numberToText(number, 4);
        return res;
    }

    void WordToNum::run(std::string &sentence, std::shared_ptr<KenLM> lm)
    {
        if (sentence.empty())
            return;
        std::vector<std::string> words;
        tokenize(sentence, ' ', words);
        int wordInd = 0;
        hyps_.clear();
        bool getBest = false;
        uint unsymCount = 0;
        if (hyps_.size() < words.size())
        {
            for (int i = hyps_.size(); i < words.size(); i++)
            {
                hyps_.emplace(i, std::vector<HypsState>());
            }
        }
        hyps_[0].emplace_back("", nullptr, false, true);
        for (auto word : words)
        {
            if(isSynonym(word))
            {
                unsymCount = 0;
                getBest = true;
            } else {
                unsymCount++;
            }
            for (HypsState &prevHyp : hyps_[wordInd])
            {
                if (!isSynonym(word))
                {
                    if (prevHyp.nextNum)
                    {
                        prevHyp.word += " " + greaterThousandToText(prevHyp.qNum);
                        prevHyp.nextNum = false;
                    }
                    hyps_[wordInd + 1].emplace_back(word, &prevHyp);
                } else {
                    if (!prevHyp.checked)
                        {
                            if (prevHyp.nextNum)
                            {
                                hyps_[wordInd + 1].emplace_back(greaterThousandToText(prevHyp.qNum) + " " + word, &prevHyp, true);
                            } else {
                                hyps_[wordInd + 1].emplace_back(word, &prevHyp, true);
                            }
                        }
                    prevHyp.nextNum = true;
                    prevHyp.qNum.push_back(word);
                    hyps_[wordInd + 1].emplace_back(prevHyp);
                }
            }
            if ((unsymCount > 2) && (getBest))
            {
                getBestHypothesis(hyps_[wordInd + 1], lm);
                getBest = false;
            }
            wordInd ++;
        }
        // Finish creating hypothesis
        for (HypsState &hyp : hyps_[wordInd])
        {
            if (hyp.nextNum)
            {
                hyp.word += " " + greaterThousandToText(hyp.qNum);
                hyp.nextNum = false;
            }
        }
        std::vector<std::string> hypsSentences;
        for (auto &hyp : hyps_[wordInd])
        {
            HypsState *hypPtr = &hyp;
            std::string hypSen = "";
            while (hypPtr)
            {
                hypSen = hypPtr->word + " " + hypSen;
                hypPtr = hypPtr->parents;
            }
            applyRules(hypSen);
            hypsSentences.push_back(hypSen);
        }
        // Get the best hypothesis
        sentence = *std::max_element(hypsSentences.begin(), hypsSentences.end(),
                                     [this, lm](std::string a, std::string b) {
                                         return lm->score(a) < lm->score(b);
                                     });
    }

    void WordToNum::getBestHypothesis(std::vector<HypsState> &hyps, std::shared_ptr<KenLM> lm)
    {
        std::vector<std::string> sentences;
        int len = hyps.size();
        for (auto &hyp : hyps)
        {
            HypsState *hypPtr = &hyp;
            std::string hypSen = "";
            while (hypPtr)
            {
                hypSen = hypPtr->word + " " + hypSen;
                hypPtr = hypPtr->parents;
            }
            applyRules(hypSen);
            sentences.push_back(hypSen);
        }

        int bestIndex = 0;
        float bestScore = lm->score(sentences[0]);
        for (int i = 1; i < len; i++)
        {
            float s = lm->score(sentences[i]);
            if (bestScore < s)
            {
                bestScore = s;
                bestIndex = i;
            }
        }
        HypsState bestHyp = hyps.at(bestIndex);
        hyps.clear();
        hyps.emplace_back(bestHyp);
    }
} // namespace pnk