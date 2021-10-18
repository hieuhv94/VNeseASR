/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "lm/model.hh"
#include "lm/config.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "util/string_stream.hh"
#include <memory>
namespace pnk {
class KenLM {
    public:
        KenLM(std::string lmPath) : path_(lmPath) {}
        void load();
        float score(std::string text);
    private:
        std::string path_;
        std::shared_ptr<lm::ngram::Model> lm_;
};
}