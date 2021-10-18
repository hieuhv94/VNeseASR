/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "Modules.h"
namespace pnk
{
    void AsrModules::initial(AsrModulesConfig config)
    {
        // Load feature extraction module
        featureModule = std::make_shared<streaming::Sequential>();
        {
            if (!checkFileExisted(config.featurePath))
            {
                throw std::runtime_error(std::string("[Feature] Cannot found feature extraction file at ") + config.featurePath);
                return;
            }
            std::ifstream featFile(
                config.featurePath, std::ios::binary);
            if (!featFile.is_open())
            {
                throw std::runtime_error(
                    "[Feature] Failed to open feature file=" +
                    config.featurePath + " for reading");
            }
            cereal::BinaryInputArchive ar(featFile);
            ar(featureModule);
        }

        // Load acoustic model module
        acousticModule = std::make_shared<streaming::Sequential>();
        {
            if (!checkFileExisted(config.amPath))
            {
                throw std::runtime_error(std::string("[Am Network] Cannot found acoustic model at ") + config.amPath);
                return;
            }
            std::ifstream amFile(
                config.amPath, std::ios::binary);
            if (!amFile.is_open())
            {
                throw std::runtime_error(
                    "[Am Network] Failed to open acoustic model file=" +
                    config.amPath + " for reading");
            }
            cereal::BinaryInputArchive ar(amFile);
            ar(acousticModule);
        }

        // Read tokens
        std::ifstream tknFile(config.tokensPath);
        if (!tknFile.is_open())
        {
            throw std::runtime_error(
                "[Decoder] Failed to open tokens file=" +
                config.tokensPath + " for reading");
        }
        std::string line;
        while (std::getline(tknFile, line))
        {
            tokens.push_back(line);
        }
        int nTokens = tokens.size();
        LOG(INFO) << "Tokens loaded - " << nTokens << " tokens" << std::endl;

        // Build transitions
        if (!config.transPath.empty())
        {
            std::ifstream transitionsFile(
                config.transPath, std::ios::binary);
            if (!transitionsFile.is_open())
            {
                throw std::runtime_error(
                    "[Decoder] Failed to open transition parameter file=" +
                    config.transPath + " for reading");
            }
            cereal::BinaryInputArchive ar(transitionsFile);
            ar(transitions);
        }

        // Build decoder
        decoderOptions.beamSize = config.beamSize;
        decoderOptions.beamSizeToken = config.beamSizeToken;
        decoderOptions.beamThreshold = config.beamThreshold;
        decoderOptions.lmWeight = config.lmWeight;
        decoderOptions.wordScore = config.wordScore;
        decoderOptions.unkScore = config.unknownScore;
        decoderOptions.silScore = config.silientScore;
        decoderOptions.logAdd = config.logAdd;
        decoderOptions.criterionType = config.criterionType;

        decoderFactory = std::make_shared<DecoderFactory>(
            config.tokensPath,
            config.lexiconPath,
            config.lmPath,
            transitions,
            config.smearing,
            "_",
            0);
    }

    void TextCorrectionModules::initial(TextCorrectionModulesConfig config)
    {
        kenLM = std::make_shared<KenLM>(config.lmNumPath);
        kenLM->load();

        predPunc = std::make_shared<PunctuationAndNer>(config.puncPath, config.punc, config.ner);
    }
} // namespace pnk