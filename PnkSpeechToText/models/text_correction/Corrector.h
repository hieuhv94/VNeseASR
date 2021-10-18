/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include <string>
#include <algorithm> 
#include <cctype>
#include <locale>
#include "models/text_correction/WordToNum.h"
#include "models/text_correction/punctner/PunctNer.h"
namespace pnk
{
    class PnkTextCorrector
    {
    public:
        PnkTextCorrector();
        std::string run(const std::string sequence, 
                        std::shared_ptr<KenLM> lm,
                        std::shared_ptr<PunctuationAndNer> punc);
    private:
        std::unique_ptr<WordToNum> w2n_;
        std::shared_ptr<PunctuationAndNer> predPunc_;
    };
} // namespace pnk