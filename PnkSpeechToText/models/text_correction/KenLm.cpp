/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "KenLm.h"

namespace pnk
{
    void KenLM::load()
    {
        lm::ngram::Config config;
        config.load_method = util::READ;

        lm_ = std::make_shared<lm::ngram::Model>(path_.c_str(), config);
    }

    float KenLM::score(std::string text)
    {
        float score;
        lm::ngram::State state, out_state;
        lm::FullScoreReturn ret;

        state = lm_->BeginSentenceState();
        score = 0;
        for (util::TokenIter<util::SingleCharacter, true> it(text, ' '); it; ++it)
        {
            lm::WordIndex vocab = lm_->GetVocabulary().Index(*it);
            ret = lm_->FullScore(state, vocab, out_state);
            score += ret.prob;
            state = out_state;
        }
        ret = lm_->FullScore(state, lm_->GetVocabulary().EndSentence(), out_state);
        score += ret.prob;
        return score;
    }
} // namespace pnk