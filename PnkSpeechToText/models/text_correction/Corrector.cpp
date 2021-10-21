/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "Corrector.h"

namespace pnk {
        // trim from start (in place)
    static inline void ltrim(std::string &s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
    }

    // trim from end (in place)
    static inline void rtrim(std::string &s)
    {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(),
                s.end());
    }

    // trim from both ends (in place)
    static inline void trim(std::string &s)
    {
        ltrim(s);
        rtrim(s);
    }
    
    static inline std::vector<std::string> splitString(std::string input)
    {
        std::istringstream ss(input);
        std::string token;

        std::vector<std::string> tokens;
        while(std::getline(ss, token, ' ')) {
		    tokens.push_back(token);
	    }
        return tokens;
    }
    PnkTextCorrector::PnkTextCorrector()
    {
        w2n_ = std::make_unique<WordToNum>();
    }

    std::string PnkTextCorrector::run(const std::string sequence, std::shared_ptr<KenLM> lm, std::shared_ptr<PunctuationAndNer> punc)
    {
        std::string out = sequence;
        w2n_->run(out, lm);
        trim(out);
        if (punc)
            punc->predict(out);
        return out;
    }
}