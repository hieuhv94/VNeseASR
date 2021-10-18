/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "models/asr/decoder/Decoder.h"
#include "models/asr/feature/feature.h"
#include "models/asr/nn/nn.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include "config.h"
using namespace w2l;
using namespace streaming;
namespace pnk
{   
    class AsrModules {
        public:
        AsrModules(){};
        void initial(AsrModulesConfig config);
        size_t tokensSize(){
            return tokens.size();
        }
        fl::lib::text::LexiconDecoderOptions decoderOptions;
        std::shared_ptr<streaming::Sequential> featureModule;
        std::shared_ptr<streaming::Sequential> acousticModule;
        std::vector<std::string> tokens;
        std::vector<float> transitions;
        std::shared_ptr<const DecoderFactory> decoderFactory = nullptr;
        std::shared_ptr<w2l::streaming::Decoder> decoder = nullptr;
    };

    class TextCorrectionModules {
        public:
        TextCorrectionModules(){};
        void initial(TextCorrectionModulesConfig config);
        std::shared_ptr<KenLM> kenLM;
        std::shared_ptr<PunctuationAndNer> predPunc;
    };
}